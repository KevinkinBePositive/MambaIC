from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
import torch
import torch.nn as nn
from modules import VSSBlock
from torch import Tensor
from compressai.models import CompressionModel
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from einops import rearrange 
import math

from modules import conv, deconv, ste_round, update_registered_buffers, get_scale_table

global_var = 0
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


class Quantizer():
    def quantize(self, inputs, quantize_type="noise"):
        if quantize_type == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)

class WLA(nn.Module):
        def __init__(self, input_dim, output_dim, head_dim, window_size):
            super(WLA, self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.head_dim = head_dim 
            self.scale = self.head_dim ** -0.5
            self.n_heads = input_dim//head_dim
            self.window_size = window_size
            self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
            self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

            self.linear = nn.Linear(self.input_dim, self.output_dim)

            trunc_normal_(self.relative_position_params, std=.02)
            self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

        def generate_mask(self, h, w, p, shift):
            """ generating the mask of SW-MSA
            Args:
                shift: shift parameters in CyclicShift.
            Returns:
                attn_mask: should be (1 1 w p p),
            """
            attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
            return attn_mask


        def forward(self, x):
            """ Forward pass of Window Multi-head Self-attention module.
            Args:
                x: input tensor with shape of [b h w c];b c h w
                attn_mask: attention mask, fill -inf where the value is True; 
            Returns:
                output: tensor shape [b h w c]
            """
            x = x.permute(0, 2, 3, 1)
            x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
            h_windows = x.size(1)
            w_windows = x.size(2)
            x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
            qkv = self.embedding_layer(x)
            q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
            sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
            sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
            
            probs = nn.functional.softmax(sim, dim=-1)
            output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
            output = rearrange(output, 'h b w p c -> b w p (h c)')
            output = self.linear(output)
            output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

            return output.permute(0, 3, 1, 2)

        def relative_embedding(self):
            cord = torch.tensor(
                [[i, j] for i in range(self.window_size) for j in range(self.window_size)],
                dtype=torch.long,  # 显式指定为torch.long
                device=self.relative_position_params.device
            )
            relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
            return self.relative_position_params[:, relation[:,:,0], relation[:,:,1]]

class MambaIC(CompressionModel):
    def __init__(self, depths=[2, 2, 9, 2], drop_path_rate=0.1, N=128,  M=320, num_slices=5, max_support_slices=5, **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        # 各层的 vss block 的数量
        self.depths = depths
        # 
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        dim = N
        # y的维度：320
        self.M = M
        # 丢弃率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # 分组
        group_channels = self.M // num_slices
        self.groups = [0] + [group_channels] * num_slices

        self.m_down1 = [VSSBlock(hidden_dim=2 * N, drop_path=dpr[i], use_checkpoint=False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2") 
                      for i in range(depths[0])] + [conv(N*2, N*2, kernel_size=3, stride=2)]
        
        self.m_down2 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i + depths[0]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[1])] + [conv(N*2, N*2, kernel_size=3, stride=2)]
        
        self.m_down3 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i + depths[0] + depths[1]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[2])] + [conv(N*2, M, kernel_size=3, stride=2)]
        
        # GA Encoder [1, 3, 256, 256] -> [1, 320, 16, 16]
        self.g_a = nn.Sequential(*[conv(3, N*2, kernel_size=5, stride=2)] + self.m_down1 + self.m_down2 + self.m_down3)

        self.ha_down1 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i + depths[0] + depths[1] + depths[2]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[3])] + [conv(N*2, 192, kernel_size=3, stride=2)]

        self.ha_down2 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i + depths[0] + depths[1] + depths[2]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[3])] + [conv(N*2, 128, kernel_size=3, stride=2)]

        # h_a编码器：y -> z [1, 320, 16, 16] -> [1, 192, 4, 4]
        self.h_a = nn.Sequential(
            *[conv(M, N*2, kernel_size=3, stride=2)] + self.ha_down1 + [conv(192, N*2, kernel_size=3, stride=2)] + self.ha_down2)

        # 倒序
        depths = depths[::-1]

        self.hs_up1 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[0])] + [deconv(N*2, M, kernel_size=3, stride=2)]
        
        self.hs_up2 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[0])] + [deconv(N*2, M, kernel_size=3, stride=2)]
        
        self.h_mean_s = nn.Sequential(
            *[deconv(128, N*2, kernel_size=3, stride=2)] + self.hs_up1
        + [deconv(320, N*2, kernel_size=3, stride=2)] + self.hs_up2)

        self.hs_up21 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2") 
                      for i in range(depths[0])] + [deconv(N*2, M, kernel_size=3, stride=2)]

        self.hs_up22 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2") 
                      for i in range(depths[0])] + [deconv(N*2, M, kernel_size=3, stride=2)]

        self.h_scale_s = nn.Sequential(
            *[deconv(128, N*2, kernel_size=3, stride=2)] + self.hs_up21
        + [deconv(320, N*2, kernel_size=3, stride=2)] + self.hs_up22)

        # Gs Decoder
        self.m_up1 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i + depths[0]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[1])] + [deconv(N*2, N*2, kernel_size=3, stride=2)]
        
        self.m_up2 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i + depths[0] + depths[1]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                      for i in range(depths[2])] + [deconv(N*2, N*2, kernel_size=3, stride=2)]
        
        self.m_up3 = [VSSBlock(hidden_dim = 2*N, drop_path = dpr[i + depths[0] + depths[1] + depths[2]], use_checkpoint = False, 
                                 norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2") 
                      for i in range(depths[3])] + [deconv(N*2, 3, kernel_size=3, stride=2)]

        self.g_s = nn.Sequential(*[deconv(M, N*2, kernel_size=5, stride=2)] + self.m_up1 + self.m_up2 + self.m_up3)


        # 熵编码/解码 AE/AD
        self.entropy_bottleneck = EntropyBottleneck(128)
        self.gaussian_conditional = GaussianConditional(None)

        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                *[VSSBlock(hidden_dim=self.groups[min(1, i) if i > 0 else 0] + self.groups[i if i > 1 else 0], 
                           drop_path=dpr[j], use_checkpoint=False, 
                           norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                           ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2") 
                  for j in range(depths[0])],
                conv(self.groups[min(1, i) if i > 0 else 0] + self.groups[i if i > 1 else 0], self.groups[i + 1]*2, kernel_size=3, stride=1),
            ) for i in range(1,  num_slices)
        )

        self.quantizer = Quantizer()

        self.ParamAggregation = nn.ModuleList(
        WLA(640 + self.groups[i+1 if i > 0 else 0] * 2 + self.groups[
                    i + 1] * 2, self.groups[i + 1]*2, 32, 8) for i in range(5)
        )

        self.context_prediction = nn.ModuleList(
            nn.Sequential(
                *[VSSBlock(hidden_dim=self.groups[i+1], 
                            drop_path=dpr[j], use_checkpoint=False, 
                            norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", 
                            ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2")
                            for j in range(depths[0])],
                nn.Conv2d(self.groups[i+1], 2*self.groups[i+1], kernel_size=5, padding=2, stride=1),
            ) for i in range(5)
        )

    
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


    def forward(self, x):
        
        y = self.g_a(x)
        B, C, H, W = y.size()
        
        y_shape = y.shape[2:]
        
        z = self.h_a(y)

        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        anchor = torch.zeros_like(y).to(x.device)
        non_anchor = torch.zeros_like(y).to(x.device)

        anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        y_slices = torch.split(y, self.groups[1:], 1)

        anchor_split = torch.split(anchor, self.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device),
                                              [2 * i for i in self.groups[1:]], 1)
        y_hat_slices = []
        y_hat_slices_for_gs = []
        y_likelihood = []
        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index-1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index-1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index-1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            
            y_anchor = anchor_split[slice_index]
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            scales_hat_split = torch.zeros_like(y_anchor).to(x.device)
            means_hat_split = torch.zeros_like(y_anchor).to(x.device)

            scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]

            y_anchor_quantilized = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor
            y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor


            y_anchor_quantilized[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized[:, :, 1::2, 0::2] = 0
            y_anchor_quantilized_for_gs[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized_for_gs[:, :, 1::2, 0::2] = 0

            
            masked_context = self.context_prediction[slice_index](y_anchor_quantilized)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
            # 熵模型
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat_split, means=means_hat_split)

            y_non_anchor = non_anchor_split[slice_index]
    
            y_non_anchor_quantilized = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                                        "ste") + means_non_anchor
            y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                                        "ste") + means_non_anchor

            y_non_anchor_quantilized[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized[:, :, 1::2, 1::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 1::2, 1::2] = 0

            y_hat_slice = y_anchor_quantilized + y_non_anchor_quantilized
            y_hat_slice_for_gs = y_anchor_quantilized_for_gs + y_non_anchor_quantilized_for_gs
            y_hat_slices.append(y_hat_slice)
            
            y_hat_slices_for_gs.append(y_hat_slice_for_gs)
            y_likelihood.append(y_slice_likelihood)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        """
        use STE(y) as the input of synthesizer
        """
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # pdb.set_trace()
        super().load_state_dict(state_dict)


    def compress(self, x):
        # representation y
        import time
        y_enc_start = time.time()
        y = self.g_a(x)
        y_enc = time-time() - y_enc_start
        B, C, H, W = y.size()
        y_shape = y.shape[2:]

        z_enc_start = time.time()
        z = self.h_a(y)
        z_enc = time.time() - z_enc_start
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        z_dec_start = time.time()
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        z_dec = time.time() - z_dec_start

        y_slices = torch.split(y, self.groups[1:], 1)

        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device), [2 * i for i in self.groups[1:]], 1)

        y_strings = []
        y_hat_slices = []
        params_start = time.time()
        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)

            y_anchor = y_slices[slice_index].clone()
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = y_anchor.size()

            y_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor//2).to(x.device)
            means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor//2).to(x.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(x.device)

            y_anchor_encode[:, :, 0::2, :] = y_anchor[:, :, 0::2, 0::2]
            y_anchor_encode[:, :, 1::2, :] = y_anchor[:, :, 1::2, 1::2]
            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = self.gaussian_conditional.compress(y_anchor_encode, indexes_anchor, means=means_anchor_encode)
            anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor, means=means_anchor_encode)
            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]



            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            y_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(x.device)

            non_anchor = y_slices[slice_index].clone()
            y_non_anchor_encode[:, :, 0::2, :] = non_anchor[:, :, 0::2, 1::2]
            y_non_anchor_encode[:, :, 1::2, :] = non_anchor[:, :, 1::2, 0::2]
            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]

            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
            non_anchor_strings = self.gaussian_conditional.compress(y_non_anchor_encode, indexes_non_anchor,
                                                                    means=means_non_anchor_encode)

            non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
                                                                        means=means_non_anchor_encode)

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_slice_hat = y_anchor_decode + y_non_anchor_quantized
            y_hat_slices.append(y_slice_hat)

            y_strings.append([anchor_strings, non_anchor_strings])

        params_time = time.time() - params_start
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],
                "time": {'y_enc': y_enc, "z_enc": z_enc, "z_dec": z_dec, "params": params_time}}
    
    
    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        B, _, _, _ = z_hat.size()

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_strings = strings[0]

        ctx_params_anchor = torch.zeros((B, self.M*2, z_hat.shape[2] * 4, z_hat.shape[3] * 4)).to(z_hat.device)
        ctx_params_anchor_split = torch.split(ctx_params_anchor, [2 * i for i in self.groups[1:]], 1)


        y_hat_slices = []
        for slice_index in range(len(self.groups) - 1):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)

            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = means_anchor.size()

            means_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            scales_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(z_hat.device)

            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_encode)
            anchor_strings = y_strings[slice_index][0]
            anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor,
                                                                    means=means_anchor_encode)

            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]


            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            means_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)
            scales_non_anchor_encode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor // 2).to(z_hat.device)

            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 0::2]

            indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor_encode)
            non_anchor_strings = y_strings[slice_index][1]
            non_anchor_quantized = self.gaussian_conditional.decompress(non_anchor_strings, indexes_non_anchor,
                                                                        means=means_non_anchor_encode)

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_slice_hat = y_anchor_decode + y_non_anchor_quantized
            y_hat_slices.append(y_slice_hat)
        y_hat = torch.cat(y_hat_slices, dim=1)

        import time
        y_dec_start = time.time()
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        y_dec = time.time() - y_dec_start

        return {"x_hat": x_hat, "time":{"y_dec": y_dec}}



if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256).to('cuda')
    model = MambaIC().cuda()
    y = model(x)
    print("shape of y:{}".format(y["x_hat"].shape))
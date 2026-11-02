import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from typing import Callable, Optional


# From mdgen (github.com/bjing2016/mdgen, mdgen/model/layers.py), MIT License, Copyright (c) Meta Platforms, Inc. and affiliates;
# the design follows DiT (Peebles and Xie, github.com/facebookresearch/DiT).
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.

    ``time_factor`` multiplies the input before the sinusoidal embedding —
    Flux pattern (https://github.com/black-forest-labs/flux). With
    ``max_period=10000`` the embedding's frequencies span ``[1e-4, 1]``, so
    it is only discriminative for inputs in roughly ``[1, max_period]``; a
    ``time_factor`` of ~1000 makes all dims discriminative for flow-matching
    times in ``[0, 1]``.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, time_factor: float = 1.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.time_factor = float(time_factor)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, time_factor: float = 1.0):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output (even).
        :param max_period: controls the minimum frequency of the embeddings.
        :param time_factor: pre-multiplies ``t`` (Flux pattern).
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = time_factor * t
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(
            t,
            self.frequency_embedding_size,
            time_factor=self.time_factor,
        )
        t_emb = self.mlp(t_freq)
        return t_emb


# Modified from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L58
# timm (github.com/huggingface/pytorch-image-models), Apache License 2.0, Copyright 2019 Ross Wightman.
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Reshape the mask for broadcasting
        attn_mask = mask.unsqueeze(1).unsqueeze(2)  # shape: [B, 1, 1, N]
        # turn the mask to bool type
        attn_mask = attn_mask.to(torch.bool)

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, attn_mask=attn_mask
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Modified from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L58
# timm (github.com/huggingface/pytorch-image-models), Apache License 2.0, Copyright 2019 Ross Wightman.
class DoubleStreamSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        last_block: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = nn.Dropout(attn_drop)

        # X Stream (Atoms)
        self.x_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.x_q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.x_k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.x_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.x_proj_drop = nn.Dropout(proj_drop)

        # Y Stream (NMR Peaks)
        self.y_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.y_q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.y_k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        if not last_block:
            self.y_proj = nn.Linear(dim, dim, bias=proj_bias)
            self.y_proj_drop = nn.Dropout(proj_drop)
        self.last_block = last_block

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        x_B, x_N, x_C = x.shape
        y_B, y_N, y_C = y.shape

        # --- Stream X (Atoms) ---
        x_qkv = self.x_qkv(x).reshape(x_B, x_N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        x_q, x_k, x_v = x_qkv.unbind(0)
        x_q, x_k = self.x_q_norm(x_q), self.x_k_norm(x_k)

        # --- Stream Y (Peaks) ---
        y_qkv = self.y_qkv(y).reshape(y_B, y_N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        y_q, y_k, y_v = y_qkv.unbind(0)
        y_q, y_k = self.y_q_norm(y_q), self.y_k_norm(y_k)

        # --- Joint Concatenation ---
        q = torch.cat([x_q, y_q], dim=2)  # shape: [B, num_heads, N_x + N_y, head_dim]
        k = torch.cat([x_k, y_k], dim=2)
        v = torch.cat([x_v, y_v], dim=2)

        # Masking: shape [B, 1, 1, N_x + N_y] -> True means "allow attention"
        mask = torch.cat([x_mask, y_mask], dim=1).bool()
        attn_mask = mask.unsqueeze(1).unsqueeze(2)

        # --- Attention Calculation ---
        tokens = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, attn_mask=attn_mask
        )

        # --- Split & Project Back ---
        tokens = tokens.transpose(1, 2).reshape(x_B, x_N + y_N, x_C)

        x_out, y_out = tokens.split([x_N, y_N], dim=1)

        x_out = self.x_proj_drop(self.x_proj(x_out))
        if not self.last_block:
            y_out = self.y_proj_drop(self.y_proj(y_out))

        return x_out, y_out


# From DINOv2 (github.com/facebookresearch/dinov2, dinov2/layers/swiglu_ffn.py), Apache License 2.0, Copyright (c) Meta Platforms, Inc. and affiliates.
class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.dropout(hidden)
        return self.w3(hidden)


# From mdgen (github.com/bjing2016/mdgen, mdgen/model/layers.py), MIT License, Copyright (c) Meta Platforms, Inc. and affiliates;
# the design follows DiT (Peebles and Xie, github.com/facebookresearch/DiT).
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DoubleStreamBlock(nn.Module):
    """
    A MMDiT (DoubleStream) block adapted from Flux.2 model.
    """

    def __init__(
        self,
        hidden_size,
        n_heads,
        mlp_ratio=4,
        last_block=False,
        use_bias=False,
    ):
        super().__init__()
        self.last_block = last_block

        # Initialize normalization layer for both streams
        self.x_norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.x_norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.y_norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if not last_block:
            self.y_norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = DoubleStreamSelfAttention(
            hidden_size,
            n_heads,
            qkv_bias=use_bias,
            last_block=last_block,
            proj_bias=use_bias,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.x_mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim), bias=use_bias)
        if not last_block:
            self.y_mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim), bias=use_bias)

        self.x_adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=use_bias)
        )
        if not last_block:
            y_adaLN_chunks = 6
        else:
            y_adaLN_chunks = 2  # No MLP modulation for the last block of Y stream
        self.y_adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, y_adaLN_chunks * hidden_size, bias=use_bias)
        )

    def forward(self, c, x, x_mask, y, y_mask):
        # 1. Modulation Parameters
        x_shift_msa, x_scale_msa, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = (
            self.x_adaLN_modulation(c).chunk(6, dim=1)
        )
        if not self.last_block:
            y_shift_msa, y_scale_msa, y_gate_msa, y_shift_mlp, y_scale_mlp, y_gate_mlp = (
                self.y_adaLN_modulation(c).chunk(6, dim=1)
            )
        else:
            y_shift_msa, y_scale_msa = self.y_adaLN_modulation(c).chunk(2, dim=1)

        # 2. Joint Attention
        x_pre_attn = modulate(self.x_norm1(x), x_shift_msa, x_scale_msa)
        y_pre_attn = modulate(self.y_norm1(y), y_shift_msa, y_scale_msa)

        x_attn, y_attn = self.attn(x_pre_attn, y_pre_attn, x_mask, y_mask)

        x = x + x_gate_msa.unsqueeze(1) * x_attn
        x = x + x_gate_mlp.unsqueeze(1) * self.x_mlp(modulate(self.x_norm2(x), x_shift_mlp, x_scale_mlp))

        if not self.last_block:
            y = y + y_gate_msa.unsqueeze(1) * y_attn
            y = y + y_gate_mlp.unsqueeze(1) * self.y_mlp(modulate(self.y_norm2(y), y_shift_mlp, y_scale_mlp))

        return x, y


# From mdgen (github.com/bjing2016/mdgen, mdgen/model/layers.py), MIT License, Copyright (c) Meta Platforms, Inc. and affiliates;
# the design follows DiT (Peebles and Xie, github.com/facebookresearch/DiT).
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_size, use_bias=False):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=use_bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=use_bias)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

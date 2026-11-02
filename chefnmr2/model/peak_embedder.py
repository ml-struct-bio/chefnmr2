import torch
import torch.nn as nn
from chefnmr2.model.layers import Attention, SwiGLUFFN


############################################################################
# Shift Encoder
############################################################################
class FourierShiftEncoder(nn.Module):
    """
    Encode ppm shift with Fourier features:
        x -> [sin(2pi f_i x), cos(2pi f_i x)] for i=1..K
    """

    def __init__(
        self,
        shift_min: float,
        shift_max: float,
        normalize_shift: bool,
        f_min: float,
        f_max: float,
        num_freqs: int,
    ):
        super().__init__()
        # log-spaced frequencies cover both coarse + fine patterns
        freqs = torch.logspace(
            torch.log10(torch.tensor(f_min)), torch.log10(torch.tensor(f_max)), steps=num_freqs
        )
        self.register_buffer("freqs", freqs)

        self.shift_min = shift_min
        self.shift_max = shift_max
        self.normalize_shift = normalize_shift

    def forward(self, x_ppm: torch.Tensor) -> torch.Tensor:
        if self.normalize_shift:
            x_ppm = (x_ppm - self.shift_min) / (self.shift_max - self.shift_min)
        # x_ppm: (B, P)
        # -> (B, P, 2K)
        x = x_ppm.unsqueeze(-1)  # (B, P, 1)
        w = self.freqs.view(1, 1, -1)  # (1,1,K)
        z = 2.0 * torch.pi * x * w
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)


#########################################################################
# CNMR Peak Embedder
#########################################################################
class CPeakEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        shift_min: float = -20.0,
        shift_max: float = 230.0,
        normalize_shift: bool = True,
        f_min: float = 1.0,
        f_max: float = 256.0,
        num_shift_freqs: int = 384,
        mlp_ratio: int = 2,
    ):
        super().__init__()
        self.shift_min = shift_min
        self.shift_max = shift_max

        self.shift_enc = FourierShiftEncoder(
            num_freqs=num_shift_freqs,
            f_min=f_min,
            f_max=f_max,
            shift_min=shift_min,
            shift_max=shift_max,
            normalize_shift=normalize_shift,
        )
        self.feature_dim = num_shift_freqs * 2

        # 2-layer MLP, then post-norm
        mlp_dim = hidden_dim * mlp_ratio
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, hidden_dim),
            Norm(hidden_dim),
        )

    def forward(self, c_peak_centroid: torch.Tensor, c_peak_mask: torch.Tensor) -> torch.Tensor:
        c_peak_mask = c_peak_mask.float()
        shift_feat = self.shift_enc(c_peak_centroid)  # (B, P, feature_dim)
        tokens = self.proj(shift_feat)  # (B, P, hidden_dim)
        tokens = tokens * c_peak_mask.unsqueeze(-1)
        return tokens


#########################################################################
# HNMR Peak Embedder
#########################################################################
class HPeakEmbedding(nn.Module):
    """
    HNMR Peak Embedding Module.
    Fuses Multi-Scale Fourier Shifts + Integral (nH) + Multiplet Type.
    """

    def __init__(
        self,
        hidden_dim: int = 768,  # The transformer's d_model
        # --- Shift (Fourier) Config ---
        shift_min: float = -2.0,
        shift_max: float = 12.0,
        normalize_shift: bool = True,  # Whether to normalize shift to [0,1] before Fourier encoding
        f_min: float = 1.0,
        f_max: float = 400.0,
        num_shift_freqs: int = 256,
        mlp_ratio: int = 2,
        shift_dim: int = 512,
        # --- nH (Integral) Config ---
        max_nH: int = 100,
        nH_emb_dim: int = 128,  # Dimension for nH embedding
        # --- Multiplet Config ---
        mp_vocab_size: int = 9,  # <pad> + 8 types (s,d,t,q,dd,dt,td,m)
        mp_dim: int = 128,  # Dimension for multiplet embedding
    ):
        super().__init__()

        # 1. Shift Encoder
        self.shift_enc = FourierShiftEncoder(
            num_freqs=num_shift_freqs,
            f_min=f_min,
            f_max=f_max,
            shift_min=shift_min,
            shift_max=shift_max,
            normalize_shift=normalize_shift,
        )

        # 1. Shift MLP
        shift_raw_dim = 2 * num_shift_freqs
        shift_hidden_dim = shift_raw_dim * mlp_ratio

        self.shift_mlp = nn.Sequential(
            nn.Linear(shift_raw_dim, shift_hidden_dim),
            nn.SiLU(),
            nn.Linear(shift_hidden_dim, shift_dim),
            Norm(shift_dim),
        )

        # 2. nH Embedding: index 0 is a learned bucket for peaks with nH < 0.5 (padded slots also
        # index 0 and are zeroed by h_peak_mask in forward()).
        self.nH_emb_dim = nH_emb_dim
        self.nH_emb = nn.Embedding(max_nH + 1, nH_emb_dim)

        # 3. Multiplet Embedding
        self.mp_dim = mp_dim
        self.mp_emb = nn.Embedding(mp_vocab_size, mp_dim, padding_idx=0)

        # 4. Final Fusion Projector
        # Calculates concatenation size
        cat_dim = shift_dim + nH_emb_dim + mp_dim

        # Projects concatenated features to Transformer dimension
        self.proj = nn.Sequential(
            nn.Linear(cat_dim, hidden_dim * 2),  # Expand
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),  # Compress to d_model
            Norm(hidden_dim),
        )

    def forward(
        self,
        h_peak_centroid: torch.Tensor,  # (B,P)
        h_peak_nH: torch.Tensor,  # (B,P)
        h_peak_multiplet: torch.Tensor,  # (B,P)
        h_peak_mask: torch.Tensor,  # (B,P) bool/{0,1}
    ) -> torch.Tensor:

        # Ensure mask is boolean
        mask = h_peak_mask.bool()

        # --- A. Shift Branch ---
        # 1. Fourier Encode -> (B, P, 2*freqs)
        shift_ff = self.shift_enc(h_peak_centroid)
        # 2. MLP Project -> (B, P, shift_dim)
        shift_feat = self.shift_mlp(shift_ff)
        feats = [shift_feat]

        # --- B. nH Branch ---
        # nH is rounded to the nearest whole proton (fractional integrals are real in experimental
        # spectra), then clamped to the vocabulary to prevent index errors on huge integrals.
        nH_val = h_peak_nH.round()
        nH_idx = torch.clamp(nH_val.long(), min=0, max=self.nH_emb.num_embeddings - 1)
        feats.append(self.nH_emb(nH_idx))

        # --- C. Multiplet Branch ---
        # 0 is padding; input should be valid indices
        feats.append(self.mp_emb(h_peak_multiplet.long()))

        # --- D. Fusion ---
        # Concat: (B, P, cat_dim)
        x = torch.cat(feats, dim=-1)

        # Project: (B, P, hidden_dim)
        tokens = self.proj(x)

        # Apply Mask (Hard zeroing of padding tokens)
        tokens = tokens * mask.unsqueeze(-1)

        return tokens


##########################################################################
# Transformer Encoder and Pooling
##########################################################################


class Norm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        mlp_ratio: int,
        dropout: float,
        use_bias: bool = False,
    ):
        super().__init__()

        self.norm1 = Norm(hidden_size)
        self.norm2 = Norm(hidden_size)

        self.attn = Attention(
            dim=hidden_size,
            num_heads=n_heads,
            qkv_bias=use_bias,
            proj_bias=use_bias,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim), drop=dropout)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        x = x * mask.unsqueeze(-1)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        depth: int,
        n_heads: int,
        mlp_ratio: int,
        dropout: float,
        use_bias: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_bias=use_bias,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, m):
        for layer in self.layers:
            x = layer(x, m)
        return x


class AttnPoolToken(nn.Module):
    """
    Attention pooling to get a single vector from a sequence of vectors: a learned token attends
    to the sequence together with it.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float,
        use_bias: bool = False,
    ):
        super().__init__()
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=use_bias,
            proj_bias=use_bias,
        )

        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.proj = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x, mask):
        cls = self.cls.expand(x.size(0), -1, -1)  # (B, 1, dim)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, dim)
        cls_mask = torch.ones(x.size(0), 1, device=x.device, dtype=mask.dtype)
        mask = torch.cat([cls_mask, mask], dim=1)  # (B, N+1)
        x = self.attn(x, mask)[:, 0]  # (B, dim)
        return self.proj(x)  # (B, out_dim)


########################################################################
# Main Peak Embedder Module
########################################################################


class NMRPeakEmbedder(nn.Module):
    """
    Expects `condition` dict with keys:
        h_peak_centroid:  (B, H)
        h_peak_nH:        (B, H)
        h_peak_multiplet: (B, H)
        h_peak_mask:      (B, H)

        c_peak_centroid:  (B, C)
        c_peak_mask:      (B, C)

    Returns (out, tokens, mask): a pooled vector per molecule, one token per peak (1H block, then
    13C block) and the peak mask.
    """

    def __init__(
        self,
        feature_args: dict,
        hidden_dim: int = 768,
        output_dim: int = 768,
        dropout: float = 0.1,
        transformer_args={
            "depth": 4,
            "heads": 8,
            "mlp_ratio": 4,
            "use_bias": True,
        },
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Static-shape cache, armed only by AtomDiffusion.sample: the module's whole output
        # (constant across a denoising loop).
        self._static_shapes = False
        self._static_out = None

        self.h_embedder = HPeakEmbedding(
            hidden_dim=hidden_dim,
            shift_min=feature_args.h_peak_centroid.shift_min,
            shift_max=feature_args.h_peak_centroid.shift_max,
            normalize_shift=feature_args.h_peak_centroid.normalize_shift,
            f_min=feature_args.h_peak_centroid.f_min,
            f_max=feature_args.h_peak_centroid.f_max,
            num_shift_freqs=feature_args.h_peak_centroid.num_shift_freqs,
            mlp_ratio=feature_args.h_peak_centroid.mlp_ratio,
            shift_dim=feature_args.h_peak_centroid.shift_dim,
            max_nH=feature_args.h_peak_nH.max_nH,
            nH_emb_dim=feature_args.h_peak_nH.nH_dim,
            mp_vocab_size=feature_args.h_peak_multiplet.mp_vocab_size,
            mp_dim=feature_args.h_peak_multiplet.mp_dim,
        )

        self.c_embedder = CPeakEmbedding(
            hidden_dim=hidden_dim,
            shift_min=feature_args.c_peak_centroid.shift_min,
            shift_max=feature_args.c_peak_centroid.shift_max,
            normalize_shift=feature_args.c_peak_centroid.normalize_shift,
            f_min=feature_args.c_peak_centroid.f_min,
            f_max=feature_args.c_peak_centroid.f_max,
            num_shift_freqs=feature_args.c_peak_centroid.num_shift_freqs,
            mlp_ratio=feature_args.c_peak_centroid.mlp_ratio,
        )

        # type embeddings: 0 = H, 1 = C
        self.type_embedding = nn.Embedding(2, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # transformer encoder
        self.transformer = TransformerEncoder(
            hidden_size=hidden_dim,
            depth=transformer_args.depth,
            n_heads=transformer_args.heads,
            mlp_ratio=transformer_args.mlp_ratio,
            dropout=dropout,
            use_bias=transformer_args.use_bias,
        )

        # pooling
        self.pool = AttnPoolToken(
            dim=hidden_dim,
            out_dim=output_dim,
            num_heads=transformer_args.heads,
            dropout=dropout,
            use_bias=transformer_args.use_bias,
        )

        # project
        self.output_proj = nn.Sequential(nn.Linear(hidden_dim, output_dim), Norm(output_dim))

        # learned null embedding for the unconditional pass of classifier-free guidance (all peaks masked)
        self.null_out = nn.Parameter(torch.zeros(1, output_dim))

    def begin_static_shapes(self):
        """Arm the whole-output cache for one denoising loop (see AtomDiffusion.sample).

        The conditioning does not change across a denoising loop, so `forward(condition)` returns
        an identical `(out, tokens, mask)` on all sampling steps. Caching is exact under two
        conditions, both true of the sampling path: the module is in `eval()` (dropout off), and
        the `condition` tensors handed in are value-identical every step, which they are because
        the score model rebuilds them by `repeat_interleave` + guidance concat from the same
        source dict on every call.
        """
        self._static_shapes = True
        self._static_out = None

    def end_static_shapes(self):
        self._static_shapes = False
        self._static_out = None

    def forward(self, condition: dict) -> torch.Tensor:
        # Constant across a denoising loop; see begin_static_shapes. Returned tensors are
        # shared between steps, so nothing downstream may mutate them in place.
        if self._static_shapes and self._static_out is not None:
            return self._static_out
        out = self._forward_uncached(condition)
        if self._static_shapes and not self.training:
            self._static_out = out
        return out

    def _forward_uncached(self, condition: dict) -> torch.Tensor:
        tokens_list = []
        masks_list = []
        type_ids = []

        # H peaks
        h_peak_mask = condition["h_peak_mask"]
        h_tokens = self.h_embedder(
            condition["h_peak_centroid"],
            condition["h_peak_nH"],
            condition["h_peak_multiplet"],
            h_peak_mask,
        )  # (B, H, D)
        tokens_list.append(h_tokens)
        masks_list.append(h_peak_mask)
        type_ids.append(torch.zeros(h_tokens.size(1), dtype=torch.long, device=h_tokens.device))

        # C peaks
        c_peak_mask = condition["c_peak_mask"]
        c_tokens = self.c_embedder(condition["c_peak_centroid"], c_peak_mask)  # (B, C, D)
        tokens_list.append(c_tokens)
        masks_list.append(c_peak_mask)
        type_ids.append(torch.ones(c_tokens.size(1), dtype=torch.long, device=c_tokens.device))

        tokens = torch.cat(tokens_list, dim=1)  # (B, L, D)
        mask = torch.cat(masks_list, dim=1)  # (B, L)
        B, L, D = tokens.shape

        # type embeddings
        type_ids = torch.cat(type_ids)  # (L,)
        type_emb = self.type_embedding(type_ids)  # (L, D)
        type_emb = type_emb.unsqueeze(0).expand(B, -1, -1)  # (B, L, D)
        tokens = tokens + type_emb

        tokens = self.dropout(tokens)
        # transformer encoder
        tokens = self.transformer(tokens, mask)  # (B, L, D)

        # pooling
        out = self.pool(tokens, mask)  # (B, output_dim)
        # replace fully-masked samples (unconditional pass) with the learned null embedding
        all_masked = mask.sum(dim=1) == 0  # (B,)
        out = torch.where(all_masked.unsqueeze(-1), self.null_out.expand(B, -1), out)

        # project tokens
        tokens = self.output_proj(tokens)  # (B, L, output_dim)
        tokens = tokens * mask.unsqueeze(-1)  # Ensure padding tokens remain zero after projection
        return out, tokens, mask

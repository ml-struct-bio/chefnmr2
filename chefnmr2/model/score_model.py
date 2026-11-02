import torch
from torch import nn
from chefnmr2.model.layers import (
    TimestepEmbedder,
    DoubleStreamBlock,
    FinalLayer,
)
from chefnmr2.model.peak_embedder import NMRPeakEmbedder

# The peak masks are zeroed for the unconditional half of classifier-free guidance; every other
# condition tensor is passed unchanged to both halves.
_DROPPED_FOR_GUIDANCE = ("h_peak_mask", "c_peak_mask")


# Atom Diffusion Transformer
# The design follows NExT-Mol (github.com/acharkq/NExT-Mol) and DiT (Peebles and Xie, github.com/facebookresearch/DiT).
class DiffusionModuleTransformer(nn.Module):
    """The score model: atom tokens and NMR peak tokens in joint-attention (MMDiT) blocks, conditioned
    on the time and the pooled spectrum through adaptive normalization; it predicts the velocity of
    every atom's coordinates."""

    def __init__(
        self,
        # Input and output sizes
        in_atom_feature_size=14,  # Input atom features, e.g., atom one-hot encoding. Exclude atom coordinates.
        out_size=3,  # Output atom coordinates.
        n_blocks=12,
        n_heads=8,
        hidden_size=768,
        mlp_ratio=4.0,
        use_bias=False,
        embedder_args=None,
        time_factor: float = 1.0,  # Pre-multiplier on `times` inside TimestepEmbedder (Flux pattern).
    ):
        super().__init__()
        # Set while a sampling loop runs (see begin_static_shapes).
        self._static_shapes = False

        # Atom embedding
        # The design follows NExT-Mol (github.com/acharkq/NExT-Mol).
        # +3 for the atom coords (r_noisy).
        self.x_embedder = nn.Sequential(
            nn.Linear(in_atom_feature_size + 3, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        # Time (noise) embedding
        self.t_embedder = TimestepEmbedder(hidden_size, time_factor=time_factor)

        peak_embedder_args = embedder_args["NMRPeakEmbedder"]
        self.y_embedder = NMRPeakEmbedder(
            feature_args=peak_embedder_args["feature_args"],
            hidden_dim=peak_embedder_args["hidden_dim"],
            output_dim=hidden_size,
            dropout=peak_embedder_args["dropout"],
            transformer_args=peak_embedder_args["transformer_args"],
        )

        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(
                DoubleStreamBlock(
                    hidden_size,
                    n_heads,
                    mlp_ratio=mlp_ratio,
                    last_block=(i == n_blocks - 1),  # Avoid unused parameters in the last block
                    use_bias=use_bias,
                )
            )

        self.final_layer = FinalLayer(hidden_size, out_size, use_bias=use_bias)

    def begin_static_shapes(self):
        """Declare that every following forward sees the SAME conditioning, so the peak embedder
        can compute its output once per sampling loop instead of once per step.

        Only `AtomDiffusion.sample` arms this, for the duration of one denoising loop, and releases
        it in a `finally`. NOT re-entrant: arming twice raises instead of letting two loops serve
        each other's cache.
        """
        if self._static_shapes:
            raise RuntimeError(
                "static-shape cache is already armed on this score model: two sampling loops "
                "are sharing one module. Sample sequentially, or give each its own model."
            )
        self._static_shapes = True
        self.y_embedder.begin_static_shapes()

    def end_static_shapes(self):
        self._static_shapes = False
        self.y_embedder.end_static_shapes()

    def forward(self, r_noisy, times, model_inputs, multiplicity=1, guidance_scale=0.0):
        # Expand per multiplicity.
        atom_mask = (
            model_inputs["atom_mask"].repeat_interleave(multiplicity, 0).bool()
        )  # (B, n_atoms), B = batch size * multiplicity
        padded_atom_mask = atom_mask[..., None]  # (B, n_atoms, 1)

        atom_features = model_inputs["atom_one_hot"].repeat_interleave(multiplicity, 0)

        condition = {}
        for key in model_inputs["condition"]:
            condition[key] = model_inputs["condition"][key].repeat_interleave(multiplicity, 0)

        if guidance_scale != 0.0:  # Classifier-free guidance
            # For guidance, we need to duplicate the input and condition.
            r_noisy = torch.cat([r_noisy, r_noisy], dim=0)
            times = torch.cat([times, times], dim=0)
            atom_mask = torch.cat([atom_mask, atom_mask], dim=0)
            padded_atom_mask = torch.cat([padded_atom_mask, padded_atom_mask], dim=0)
            atom_features = torch.cat([atom_features, atom_features], dim=0)
            for key in condition:
                if key in _DROPPED_FOR_GUIDANCE:
                    condition[key] = torch.cat([condition[key], torch.zeros_like(condition[key])], dim=0)
                else:
                    condition[key] = torch.cat([condition[key], condition[key]], dim=0)

        x = torch.cat([r_noisy, atom_features], dim=-1)  # (B, n_atoms, in_atom_feature_size + 3)
        x = self.x_embedder(x) * padded_atom_mask
        t = self.t_embedder(times)  # (B, hidden_size)

        global_cond, y, y_mask = self.y_embedder(condition)

        c = t + global_cond
        for block in self.blocks:
            x, y = block(c, x, atom_mask, y=y, y_mask=y_mask)

        x_coords = self.final_layer(x, c)  # (B, n_atoms, out_size=3)
        x_coords = x_coords * padded_atom_mask

        if guidance_scale != 0.0:  # Classifier-free guidance
            # Split the output back into two parts: guided and unconditioned.
            x_guided = x_coords[: x_coords.shape[0] // 2]
            x_unconditioned = x_coords[x_coords.shape[0] // 2 :]
            # Apply guidance scale
            x_coords = (1 + guidance_scale) * x_guided - guidance_scale * x_unconditioned

        return dict(r_update=x_coords)

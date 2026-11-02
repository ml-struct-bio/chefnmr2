"""The inference model: ChefNMR2's network built from a checkpoint's config.

`build_model(cfg)` makes the `AtomDiffusion` under the attribute `model`, so the state dict keys
(`model.*`) are those of the released `model.pt` and a strict load passes.
"""

import torch

from chefnmr2.model.diffusion import AtomDiffusion


class InferenceModel(torch.nn.Module):
    """`model` (`AtomDiffusion`) and the config it was built from (`cfg`)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n_atom_types = len(cfg.dataset_args.atom_decoder)
        score_model_args = cfg.score_model_args
        in_size = score_model_args[score_model_args.model_name].in_atom_feature_size
        if in_size != n_atom_types:
            raise ValueError(
                f"score_model_args.{score_model_args.model_name}.in_atom_feature_size is {in_size}, "
                f"but dataset_args.atom_decoder has {n_atom_types} elements"
            )
        self.model = AtomDiffusion(
            score_model_args=score_model_args, embedder_args=cfg.embedder_args, **cfg.diffusion_process_args
        )


def build_model(cfg) -> InferenceModel:
    return InferenceModel(cfg)

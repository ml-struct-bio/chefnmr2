"""Centering of atom coordinates."""

import torch


# Started from Boltz's `center_random_augmentation` (github.com/jwohlwend/boltz, src/boltz/model/modules/utils.py),
# MIT License, Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro; centering only.
def center(atom_coords, atom_mask):
    """Subtract the mean position of the real atoms (`atom_mask`) from every atom's coordinates."""
    atom_mean = torch.sum(atom_coords * atom_mask[:, :, None], dim=1, keepdim=True) / torch.sum(
        atom_mask[:, :, None], dim=1, keepdim=True
    )
    atom_coords = atom_coords - atom_mean
    return atom_coords

"""Whether two SMILES are the same molecule: the rule that judges a prediction correct."""

import rdkit.Chem as Chem

from chefnmr2.chem.smiles import canonicalize


def match_smiles(
    target_smiles: str,
    reconstructed_smiles: str,
    remove_stereo: bool = True,
) -> bool:
    """
    Whether a predicted structure is the target: the rule by which the paper counts a prediction
    correct (matching accuracy).

    Both SMILES are canonicalized without stereochemistry and compared, so two SMILES of the same
    planar structure match. With `remove_stereo=False` the target's stereochemistry must also be found
    in the prediction. A SMILES that RDKit cannot read never matches.

    Args:
        target_smiles (str): Ground truth SMILES.
        reconstructed_smiles (str): Predicted SMILES.
        remove_stereo (bool): If True, ignores stereochemistry during matching.

    Returns:
        bool: True if molecules match, False otherwise.
    """
    try:
        target_skeleton_smiles = canonicalize(target_smiles, remove_stereo=True)
        reconstructed_skeleton_smiles = canonicalize(reconstructed_smiles, remove_stereo=True)

        if target_skeleton_smiles != reconstructed_skeleton_smiles:
            return False

        if remove_stereo:
            return True
        else:
            # check full match including stereochemistry
            mol_target = Chem.MolFromSmiles(target_smiles)
            mol_reconstructed = Chem.MolFromSmiles(reconstructed_smiles)
            if mol_reconstructed.HasSubstructMatch(mol_target, useChirality=True):
                return True
    except Exception:
        return False
    return False

"""SMILES helpers: canonical SMILES."""

from rdkit import Chem


def canonicalize(
    smiles: str,
    remove_stereo: bool = False,
) -> str | None:
    """Canonicalize a SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        isomeric_smiles = not remove_stereo
        smiles_out = Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles)
        if smiles_out is None:
            return None
        return Chem.CanonSmiles(smiles_out)
    except Exception as e:
        # print(f"Error canonicalizing SMILES {smiles}: {e}")
        return None

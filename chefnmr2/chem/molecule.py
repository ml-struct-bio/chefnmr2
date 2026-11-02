"""A molecule as atoms and 3D coordinates, exported to RDKit (`Molecule`)."""

from typing import Dict, List, Union

import numpy as np
from rdkit import Chem


class Molecule:
    """
    A class representing a molecule in 3D space.

    Holds numpy arrays; handles masking and export to RDKit.
    """

    def __init__(
        self,
        atom_coords: np.ndarray,
        # Atom types (one-hot), mask and decoder:
        atom_one_hot: np.ndarray,
        atom_mask: np.ndarray,
        atom_decoder: Union[Dict[int, str], List[str]],
        remove_h: bool,
        collapse: bool = False,
    ):
        """
        Initialize the Molecule.

        Args:
            atom_coords: Coordinates of atoms. Shape (n_atoms, 3).
            atom_one_hot: One-hot encoding of atom types. Shape (n_atoms, n_atom_types).
            atom_mask: Boolean mask indicating valid atoms. Shape (n_atoms,).
            atom_decoder: Mapping from integer index to atom symbol (e.g., ['C', 'H']).
            remove_h: Whether hydrogens are considered removed (affects validation).
            collapse: If True, immediately filters out masked atoms.
        """
        # Handle atom coordinates
        if isinstance(atom_coords, np.ndarray):
            self.atom_coords = atom_coords.astype(np.float64)
        else:
            raise TypeError("atom_coords must be a numpy.ndarray")

        self.atom_coords = self.atom_coords.reshape(-1, 3)  # Ensure shape is (n_atoms, 3)

        # Handle atom types
        self.atom_one_hot = atom_one_hot
        self.atom_types = np.argmax(atom_one_hot, axis=1).astype(int)

        # Handle decoder
        self.atom_decoder = list(atom_decoder)

        # Handle mask
        if isinstance(atom_mask, np.ndarray):
            self.atom_mask = atom_mask.astype(bool)
        else:
            raise TypeError("atom_mask must be a numpy.ndarray")

        self.remove_h = remove_h

        if collapse:
            self.collapse()

    def collapse(self):
        """
        Filters the molecule data to remove masked (invalid) atoms.
        Modifies the object in-place.
        """
        # Calculate number of valid atoms
        n_atoms = np.sum(self.atom_mask)

        # Apply mask
        self.atom_coords = self.atom_coords[self.atom_mask] if self.atom_coords is not None else None
        self.atom_one_hot = self.atom_one_hot[self.atom_mask] if self.atom_one_hot is not None else None

        self.atom_types = self.atom_types[self.atom_mask].astype(int) if self.atom_types is not None else None

        # Reset mask to all True since we've collapsed
        self.atom_mask = np.ones(int(n_atoms), dtype=bool)

    def to_rdkit_molecule(self) -> Chem.Mol:
        """
        Converts the Molecule object to an RDKit molecule.

        Note: This creates a molecule with atoms and 3D coordinates but NO bonds.

        Returns:
            Chem.Mol: The RDKit molecule object.
        """
        mol = Chem.RWMol()

        # Add atoms
        for atom_idx in self.atom_types:
            atom_symbol = self.atom_decoder[atom_idx]
            a = Chem.Atom(atom_symbol)
            mol.AddAtom(a)

        # Add 3D coordinates to atoms
        # Ensure coordinates are numpy float64 for RDKit
        coords = self.atom_coords

        conf = Chem.Conformer(len(coords))
        for i, pos in enumerate(coords):
            # pos must be float64, not float32
            # float32 will raise an error: cannot extract desired type from sequence
            conf.SetAtomPosition(i, pos.astype(np.float64))

        mol.AddConformer(conf)
        return mol

"""Bond reconstruction: 3D coordinates to bonds and SMILES, with the repair chain."""

import atexit
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
from rdkit import RDLogger, Chem
from rdkit.Chem import rdDetermineBonds, rdMolDescriptors

from chefnmr2.chem.molecule import Molecule


def normalize_charged_representation(mol: Chem.Mol) -> Chem.Mol:
    """
    Convert internal charge separations back to neutral double-bond forms.

    Specifically handles:
    - Sulfoxides: [S+]([O-]) -> S(=O)

    This fixes the mismatch between RDKit's DetermineBonds output and canonical
    SMILES. RDKit reconstructs sulfoxides as charged sulfonium ions because the
    valence table lacks valence 4 for sulfur. NOTE: could fix within rdkit

    Args:
        mol: RDKit molecule with potential [S+][O-] patterns

    Returns:
        Molecule with sulfoxides normalized to S=O form
    """
    if mol is None:
        return None

    # Pattern: S+ single-bonded to O-
    # [S+;X3] = Sulfur with +1 charge and 3 connections (typical for sulfoxide)
    # [O-;X1] = Oxygen with -1 charge and 1 connection (terminal)
    pattern = Chem.MolFromSmarts("[S+;X3:1]-[O-;X1:2]")

    if not mol.HasSubstructMatch(pattern):
        return mol

    try:
        rw_mol = Chem.RWMol(mol)
        matches = mol.GetSubstructMatches(pattern)

        for s_idx, o_idx in matches:
            s_atom = rw_mol.GetAtomWithIdx(s_idx)
            o_atom = rw_mol.GetAtomWithIdx(o_idx)

            # Only convert if S has exactly +1 and O has exactly -1
            if s_atom.GetFormalCharge() == 1 and o_atom.GetFormalCharge() == -1:
                # Remove charges
                s_atom.SetFormalCharge(0)
                o_atom.SetFormalCharge(0)

                # Change single bond to double bond
                bond = rw_mol.GetBondBetweenAtoms(s_idx, o_idx)
                bond.SetBondType(Chem.BondType.DOUBLE)

        # Sanitize to update valences and aromaticity
        result = rw_mol.GetMol()
        Chem.SanitizeMol(result)
        return result

    except Exception:
        return mol


# ---------------------------------------------------------------------------------------
# Resonance / hypervalence repair for DetermineBonds output.
#
# DetermineBonds does not return BROKEN molecules -- it returns VALID BUT WRONG ones. A
# thioester S-C(=O) comes back as the charge-separated resonance form [S+]=C-[O-]: perfectly
# valence-legal, so the algorithm has no reason to reject it, just the wrong member of a
# resonance pair. The repairs below all do the same thing -- find where charge or valence got
# parked in the wrong place and walk it back along a bond path.
#
# Every rewrite is validated by THREE guards, and all three are load-bearing:
#   1. SanitizeMol      -- rejects chemically invalid results
#   2. formula conserved -- a bond shift moves bonds only; anything that changes the formula
#                          is a different molecule
#   3. no new radicals   -- THE critical one. SanitizeMol checks LEGALITY, not IDENTITY:
#                          neutralizing an azide [N-]=[N+]=N-R yields the nitrene [N]N=N-R,
#                          which has the SAME FORMULA and sanitizes happily. Only the radical
#                          count (0 -> 2) catches it. Without this guard the rules destroy
#                          azides (-39.7pp) and other charged neutrals (-51.3pp).
#
# ---------------------------------------------------------------------------------------

_BOND_ORDER = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}
_ORDER_TO_BOND = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
_MAX_SHIFT_PATH = 10  # swept: 6 -> 70.4%, 8 -> 79.8%, 10 -> 81.9%, 12 -> 81.9% (saturates)
_CHALCOGEN = ("S", "Se")
_TRIPLE_OXO = Chem.MolFromSmarts("[#16,#34:1]#[O+:2]")


def _mol_formula(mol: Chem.Mol) -> Optional[str]:
    try:
        m = Chem.Mol(mol)
        m.UpdatePropertyCache(strict=False)
        return rdMolDescriptors.CalcMolFormula(m)
    except Exception:
        return None


def _n_radicals(mol: Chem.Mol) -> Optional[int]:
    try:
        m = Chem.Mol(mol)
        m.UpdatePropertyCache(strict=False)
        return sum(a.GetNumRadicalElectrons() for a in m.GetAtoms())
    except Exception:
        return None


def _accept(cand: Chem.Mol, ref_formula: Optional[str], ref_radicals: int) -> bool:
    """The three guards. See the module note above -- the radical check is not optional."""
    try:
        Chem.SanitizeMol(cand)
    except Exception:
        return False
    # cand is now sanitized in place (fresh property cache + radicals assigned), so read it
    # directly -- _mol_formula/_n_radicals each make a defensive Chem.Mol copy that is only
    # needed for their other, UN-sanitized callers.
    if rdMolDescriptors.CalcMolFormula(cand) != ref_formula:
        return False
    n_rad = sum(a.GetNumRadicalElectrons() for a in cand.GetAtoms())
    return n_rad <= ref_radicals


def _simple_paths(mol: Chem.Mol, start: int, end: int, max_len: int = _MAX_SHIFT_PATH):
    """All simple atom paths start->end, shortest first.

    Not GetShortestPath: the shortest path is not always the CONJUGATED one, which makes
    aromatic through-ring cases unfixable.
    """
    out, stack = [], [(start, [start])]
    while stack:
        cur, path = stack.pop()
        if cur == end:
            out.append(path)
            continue
        if len(path) >= max_len:
            continue
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            i = nb.GetIdx()
            if i not in path:
                stack.append((i, path + [i]))
    out.sort(key=len)
    return out


def _flip_alternating_path(rw: Chem.RWMol, path: List[int]) -> bool:
    """Flip each bond along path by +-1. False if the path is not strictly alternating.

    The path may START high ([S+]=C-[O-], orders 2,1) or low ([S+]-N=C-[O-], orders 1,2,1):
    both flips conserve valence at every internal atom, so do NOT require a multiple bond at
    the cation end -- that guard silently rejected every 4-atom motif.
    """
    bonds = []
    for a, b in zip(path[:-1], path[1:]):
        bd = rw.GetBondBetweenAtoms(a, b)
        if bd is None or bd.GetBondType() not in _BOND_ORDER:
            return False
        bonds.append(bd)
    orders = [_BOND_ORDER[b.GetBondType()] for b in bonds]
    for i in range(len(orders) - 1):
        if (orders[i] > 1) == (orders[i + 1] > 1):
            return False
    for bd, o in zip(bonds, orders):
        bd.SetBondType(_ORDER_TO_BOND[o - 1 if o > 1 else o + 1])
    return True


def normalize_mesomeric(mol: Chem.Mol, max_pairs: int = 8) -> Optional[Chem.Mol]:
    """Neutralize +1/-1 pairs joined by an alternating bond path (a mesomeric shift).

    Subsumes normalize_charged_representation (sulfoxide is the 2-atom case) and handles
    motifs no SMARTS was written for: thioester [S+]=C-[O-], sulfilimine [S+]-N=C-[O-],
    aromatic through-ring, para-quinoid (which is why the path limit is 10, not 6).

    Deliberately +-1 only.
    """
    if mol is None:
        return None
    # Fast no-op: a mesomeric shift needs a +1/-1 charge PAIR. With no non-zero formal
    # charge the loop below would just return best==mol anyway, so skip its per-iteration
    # RWMol/Kekulize/formula/radical setup (this runs twice per molecule via
    # normalize_reconstruction).
    if not any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
        return mol
    best = mol
    for _ in range(max_pairs):
        work = Chem.RWMol(best)
        try:
            Chem.Kekulize(work, clearAromaticFlags=True)  # makes aromatic cases explicit
        except Exception:
            return best
        ref_formula, ref_rad = _mol_formula(work), (_n_radicals(work) or 0)
        pos = [a.GetIdx() for a in work.GetAtoms() if a.GetFormalCharge() == 1]
        neg = [a.GetIdx() for a in work.GetAtoms() if a.GetFormalCharge() == -1]
        if not pos or not neg:
            return best
        done = False
        for p in pos:
            for n in neg:
                for path in _simple_paths(work, p, n):
                    trial = Chem.RWMol(work)
                    if not _flip_alternating_path(trial, list(path)):
                        continue
                    trial.GetAtomWithIdx(p).SetFormalCharge(0)
                    trial.GetAtomWithIdx(n).SetFormalCharge(0)
                    cand = trial.GetMol()
                    if not _accept(cand, ref_formula, ref_rad):
                        continue
                    best, done = cand, True
                    break
                if done:
                    break
            if done:
                break
        if not done:
            return best
    return best


def _hypervalent_sites(rw: Chem.RWMol) -> List[int]:
    """Neutral chalcogens with 2 connections but total bond order 4 (the =S= artifact)."""
    out = []
    for a in rw.GetAtoms():
        if a.GetSymbol() not in _CHALCOGEN or a.GetDegree() != 2:
            continue
        if a.GetFormalCharge() != 0:
            continue
        if sum(_BOND_ORDER.get(b.GetBondType(), 0) for b in a.GetBonds()) == 4:
            out.append(a.GetIdx())
    return out


def normalize_hypervalent(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Repair =S= centres: thiophene c1ccsc1 comes back as C1=CC=S=C1 (valence-4 S on 2
    connections). It is NEUTRAL, so normalize_mesomeric never fires on it.

    Drop both S bonds to single, mark the neighbours NoImplicit so the deficit becomes
    RADICALS rather than silent implicit H, then shift the radical pair around the ring --
    the same alternating flip, between radical endpoints instead of charge endpoints.
    """
    if mol is None:
        return None
    try:
        probe = Chem.RWMol(mol)
        Chem.Kekulize(probe, clearAromaticFlags=True)
    except Exception:
        return mol
    sites = _hypervalent_sites(probe)
    if not sites:
        return mol
    ref_formula, ref_rad = _mol_formula(probe), (_n_radicals(probe) or 0)
    best = mol
    for s_idx in sites:
        work = Chem.RWMol(probe)
        atom = work.GetAtomWithIdx(s_idx)
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(nbrs) != 2:
            continue
        for b in atom.GetBonds():
            if b.GetBondType() == Chem.BondType.DOUBLE:
                b.SetBondType(Chem.BondType.SINGLE)
        for i in nbrs:
            work.GetAtomWithIdx(i).SetNoImplicit(True)
        for path in _simple_paths(work, nbrs[0], nbrs[1]):
            if s_idx in path:
                continue
            trial = Chem.RWMol(work)
            if not _flip_alternating_path(trial, list(path)):
                continue
            for i in nbrs:
                trial.GetAtomWithIdx(i).SetNoImplicit(False)
            cand = trial.GetMol()
            if not _accept(cand, ref_formula, ref_rad):
                continue
            best = cand
            probe = Chem.RWMol(best)
            try:
                Chem.Kekulize(probe, clearAromaticFlags=True)
            except Exception:
                return best
            break
    return best


def normalize_sulfoxide(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Repair R-S(#[O+])=R -> R-S(=O)-R.

    A plain sulfoxide S=O comes back as S#[O+] -- sulfur triple-bonded to an oxygen CATION,
    valence 6 where truth is 4. normalize_hypervalent only covers degree-2/order-4 chalcogen,
    and the mesomeric path starts O+#S=C (orders 3,2 -- two multiple bonds in a row) so the
    alternation check rejects it. Chemotype: sulfoxide beta-lactams and alkaloids.
    """
    if mol is None or _TRIPLE_OXO is None:
        return mol
    try:
        probe = Chem.RWMol(mol)
        Chem.Kekulize(probe, clearAromaticFlags=True)
    except Exception:
        return mol
    matches = probe.GetSubstructMatches(_TRIPLE_OXO)
    if not matches:
        return mol
    ref_formula, ref_rad = _mol_formula(probe), (_n_radicals(probe) or 0)
    for s_idx, o_idx in matches:
        work = Chem.RWMol(probe)
        bond = work.GetBondBetweenAtoms(s_idx, o_idx)
        if bond is None or bond.GetBondType() != Chem.BondType.TRIPLE:
            continue
        bond.SetBondType(Chem.BondType.DOUBLE)  # S#[O+] -> S=O  (valence 6 -> 5)
        work.GetAtomWithIdx(o_idx).SetFormalCharge(0)
        sulfur = work.GetAtomWithIdx(s_idx)
        others = [
            b
            for b in sulfur.GetBonds()
            if b.GetIdx() != bond.GetIdx() and b.GetBondType() == Chem.BondType.DOUBLE
        ]
        for bd in others:
            trial = Chem.RWMol(work)
            tb = trial.GetBondBetweenAtoms(bd.GetBeginAtomIdx(), bd.GetEndAtomIdx())
            tb.SetBondType(Chem.BondType.SINGLE)  # valence 5 -> 4; neighbour is short
            nb = bd.GetOtherAtomIdx(s_idx)
            trial.GetAtomWithIdx(nb).SetNoImplicit(True)  # deficit = radical, not silent H
            for nidx in [a.GetIdx() for a in trial.GetAtoms() if a.GetFormalCharge() == -1]:
                for path in _simple_paths(trial, nb, nidx):
                    t2 = Chem.RWMol(trial)
                    if not _flip_alternating_path(t2, list(path)):
                        continue
                    t2.GetAtomWithIdx(nidx).SetFormalCharge(0)
                    t2.GetAtomWithIdx(nb).SetNoImplicit(False)
                    cand = t2.GetMol()
                    if _accept(cand, ref_formula, ref_rad):
                        return cand
    return mol


def normalize_reconstruction(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Full repair chain, in the validated order.

    mesomeric runs twice on purpose: the hypervalent/sulfoxide repairs can expose a charge
    pair that was not shiftable before.
    """
    if mol is None:
        return None
    out = normalize_charged_representation(mol)
    out = normalize_mesomeric(out)
    out = normalize_hypervalent(out)
    out = normalize_sulfoxide(out)
    out = normalize_mesomeric(out)
    return out


def get_smiles_from_mol(mol: Chem.Mol, remove_stereo: bool) -> Tuple[Optional[str], Optional[str], str]:
    """
    Extracts canonical SMILES and the SMILES of the largest fragment from an RDKit molecule.

    Args:
        mol (Chem.Mol): The RDKit molecule object.
        remove_stereo (bool): Whether to remove stereochemistry information.

    Returns:
        Tuple[Optional[str], Optional[str], str]: A tuple containing:
            - full_smiles: The canonical SMILES of the full molecule.
            - largest_fragment_smiles: The canonical SMILES of the largest fragment (if applicable).
            - error_type: 'ok' on success, or an error category string.
            Returns (None, None, error_type) if an error occurs.
    """
    try:
        mol_no_h = Chem.RemoveHs(mol)
        if remove_stereo:
            Chem.RemoveStereochemistry(mol_no_h)

        full_smiles = Chem.MolToSmiles(mol_no_h)
        full_smiles = Chem.CanonSmiles(full_smiles)

        fragments = Chem.rdmolops.GetMolFrags(mol_no_h, asMols=True, sanitizeFrags=True)
        if len(fragments) > 1:
            largest_fragment = max(fragments, key=lambda m: m.GetNumAtoms())
            Chem.SanitizeMol(largest_fragment)
            largest_fragment_smiles = Chem.MolToSmiles(largest_fragment)
            largest_fragment_smiles = Chem.CanonSmiles(largest_fragment_smiles)
            # NOT "ok": bonds were perceived, but the result is disconnected and there is no
            # whole-molecule SMILES. Labelling it "ok" made error_counts["ok"] exceed the
            # number of usable predictions.
            return (None, largest_fragment_smiles, "fragmented")
        else:
            return (full_smiles, full_smiles, "ok")
    except Chem.rdchem.AtomValenceException:
        return (None, None, "sanitize_valence")
    except Chem.rdchem.KekulizeException:
        return (None, None, "sanitize_kekulize")
    except Exception:
        return (None, None, "sanitize_other")


def determine_bonds_with_maxiterations(
    rdkit_mol: Chem.Mol,
    max_iterations: int = 100000,
    allow_charged_fragments: bool = True,
) -> Tuple[Optional[Chem.Mol], str]:
    """
    Determine bonds using RDKit's DetermineBonds with a single (charge=0) pass, then run the
    repair chain (`normalize_reconstruction`: mesomeric / hypervalent / sulfoxide).

    Args:
        rdkit_mol: RDKit molecule with atoms and coordinates but no bonds
        max_iterations: Max iterations for DetermineBonds
        allow_charged_fragments: Allow charged fragments in result

    Returns:
        Tuple of (RDKit molecule with bonds or None, error_type string).
        error_type is 'ok' on success, or one of: 'valence', 'kekulize',
        'convergence', 'value_error', 'other_bond'.
    """
    try:
        mol_copy = Chem.RWMol(rdkit_mol)
        rdDetermineBonds.DetermineBonds(
            mol_copy,
            maxIterations=max_iterations,
            allowChargedFragments=allow_charged_fragments,
            useHueckel=False,
            charge=0,
        )
        out = normalize_reconstruction(mol_copy.GetMol())
        if out is None:
            return None, "other_bond"
        # Re-read stereo from the coordinates. The bond-order rewrites above edit the graph
        # via RWMol, which invalidates the BondDir/StereoAtoms DetermineBonds perceived, and
        # RDKit never recomputes them -- so the graph came out right but /C=C/ flipped to
        # /C=C\. The conformer is still attached, so the true stereo is right there.
        if out.GetNumConformers() > 0:
            try:
                Chem.AssignStereochemistryFrom3D(out)
            except Exception:
                pass
        return out, "ok"
    except Chem.rdchem.AtomValenceException:
        return None, "valence"
    except Chem.rdchem.KekulizeException:
        return None, "kekulize"
    except ValueError as e:
        return None, ("convergence" if "maxIterations" in str(e) else "value_error")
    except Exception:
        return None, "other_bond"


def atom_features_to_smiles_one(
    args: Tuple,
) -> Tuple[int, Optional[str], Optional[str], str]:
    """
    Worker function to process a single molecule: converts atom features to SMILES.

    Args:
        args (Tuple): A tuple containing all necessary parameters:
            (molecule_index, atom_coords, atom_one_hot, atom_mask,
             remove_h, remove_stereo, atom_decoder, max_iterations)

    Returns:
        Tuple[int, Optional[str], Optional[str], str, Optional[Chem.Mol]]:
            (molecule_index, full_smiles, largest_fragment_smiles, error_type, mol_with_bonds)
    """
    (
        molecule_index,
        atom_coords,
        atom_one_hot,
        atom_mask,
        remove_h,
        remove_stereo,
        atom_decoder,
        max_iterations,
    ) = args

    # Reconstruct the molecule object
    molecule = Molecule(
        atom_coords=atom_coords,
        atom_one_hot=atom_one_hot,
        atom_mask=atom_mask,
        atom_decoder=atom_decoder,
        remove_h=remove_h,
        collapse=True,
    )

    try:
        rdkit_mol = molecule.to_rdkit_molecule()
        mol_with_bonds, bond_error = determine_bonds_with_maxiterations(rdkit_mol, max_iterations=max_iterations)

        if mol_with_bonds:
            reconstructed_smiles, largest_fragment_smiles, smi_error = get_smiles_from_mol(
                mol_with_bonds, remove_stereo
            )
            # Keep get_smiles_from_mol's status whenever no whole-molecule SMILES was produced.
            # The old `else "ok"` clobbered the "fragmented" status (reconstructed_smiles is None
            # but largest_fragment_smiles is set) back to "ok", so error_counts["ok"] over-counted
            # disconnected reconstructions and the "fragmented" label never reached any caller.
            error_type = "ok" if reconstructed_smiles is not None else smi_error
        else:
            reconstructed_smiles = None
            largest_fragment_smiles = None
            error_type = bond_error
    except Exception:
        mol_with_bonds = None
        reconstructed_smiles = None
        largest_fragment_smiles = None
        error_type = "build_failed"

    return (
        molecule_index,
        reconstructed_smiles,
        largest_fragment_smiles,
        error_type,
        mol_with_bonds,
    )


_POOL_CACHE: Dict[int, ProcessPoolExecutor] = {}


def _get_pool(num_workers: int) -> ProcessPoolExecutor:
    """A ProcessPoolExecutor reused across calls, one per worker count.

    WHY THIS IS CACHED
    ------------------
    A pool built per call makes every call pay a fork of the parent -- which by then
    holds torch, a CUDA context and the model. On 50,000 candidates in calls of 100,
    that cost 650 s against 60 s with one pool reused: per-call pool construction,
    not chemistry, on exactly the single-molecule case of inference.

    Reuse is safe because the workers are stateless: `atom_features_to_smiles_one`
    is a pure function of its argument tuple and keeps nothing between calls.
    """
    pool = _POOL_CACHE.get(num_workers)
    if pool is None:
        pool = ProcessPoolExecutor(max_workers=num_workers)
        _POOL_CACHE[num_workers] = pool
        # wait=False: interpreter teardown should not block on in-flight RDKit.
        atexit.register(pool.shutdown, wait=False)
    return pool


def _drop_pool(num_workers: int) -> None:
    """Discard a cached pool after a worker died, so the next call rebuilds it.

    The per-call version got this for free -- a segfault in RDKit killed one
    batch's pool and the next batch made a fresh one. A cached pool that goes
    BrokenProcessPool would otherwise stay broken for the rest of the process,
    turning one bad candidate into a failed run.
    """
    pool = _POOL_CACHE.pop(num_workers, None)
    if pool is not None:
        pool.shutdown(wait=False)


def atom_features_to_smiles(
    atom_features: Dict[str, Any],
    multiplicity: int,
    remove_h: bool,
    remove_stereo: bool,
    atom_decoder: List[str],
    max_iterations: int = 100000,
    num_workers: Optional[int] = None,
) -> Tuple[List[Optional[str]], List[Optional[str]], Dict[str, int], List[str]]:
    """
    Converts a batch of atom features into SMILES strings using parallel processing.

    Args:
        atom_features (Dict[str, Any]): Dictionary containing 'atom_coords', 'atom_one_hot', 'atom_mask'.
        multiplicity (int): Number of conformers/views per molecule to process.
        remove_h (bool): Whether hydrogens were removed in the input features.
        remove_stereo (bool): Whether to strip stereochemistry from output SMILES.
        atom_decoder (List[str]): Mapping from one-hot indices to atom types.
        max_iterations (int, optional): Max iterations for DetermineBonds. Defaults to 100000.
        num_workers (int, optional): Number of parallel workers. Defaults to CPU count.

    Returns:
        Tuple[List[Optional[str]], List[Optional[str]], Dict[str, int], List[str]]:
            - List of reconstructed full SMILES.
            - List of reconstructed largest fragment SMILES.
            - Dictionary of error type counts for diagnostics.
            - The error type of each molecule, in input order.
    """

    atom_coords_list = np.array(atom_features["atom_coords"])
    atom_one_hot_list = np.array(atom_features["atom_one_hot"])
    atom_mask_list = np.array(atom_features["atom_mask"])

    # Repeat features for multiplicity (e.g., if multiple samples per molecule)
    atom_one_hot_list = np.repeat(atom_one_hot_list, multiplicity, axis=0)
    atom_mask_list = np.repeat(atom_mask_list, multiplicity, axis=0)

    # Suppress RDKit logs during bulk processing
    RDLogger.DisableLog("rdApp.*")

    # Prepare arguments for each molecule
    args_list = [
        (
            i,
            coords,
            one_hot,
            mask,
            remove_h,
            remove_stereo,
            atom_decoder,
            max_iterations,
        )
        for i, (coords, one_hot, mask) in enumerate(zip(atom_coords_list, atom_one_hot_list, atom_mask_list))
    ]

    # Determine number of workers
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    # Initialize results containers
    reconstructed_smiles_list = [None] * len(args_list)
    reconstructed_largest_fragment_smiles_list = [None] * len(args_list)
    # Per-index error type; `error_counts` aggregates it and loses the mapping.
    error_type_list = [None] * len(args_list)
    error_counts = defaultdict(int)

    # Calculate chunksize for better distribution
    chunksize = max(1, len(args_list) // (num_workers * 4))

    def _run():
        return list(
            _get_pool(num_workers).map(
                atom_features_to_smiles_one,
                args_list,
                chunksize=chunksize,
            )
        )

    try:
        results = _run()
    except BrokenProcessPool:
        # A worker died. Rebuild once and retry -- same recovery the old
        # per-call pool had implicitly. A second failure is a real error.
        _drop_pool(num_workers)
        results = _run()

    # Aggregate results
    for i, reconstructed_smiles, largest_frag_smiles, error_type, _mol_obj in results:
        reconstructed_smiles_list[i] = reconstructed_smiles
        reconstructed_largest_fragment_smiles_list[i] = largest_frag_smiles
        error_counts[error_type] += 1
        error_type_list[i] = error_type

    return (
        reconstructed_smiles_list,
        reconstructed_largest_fragment_smiles_list,
        dict(error_counts),
        error_type_list,
    )

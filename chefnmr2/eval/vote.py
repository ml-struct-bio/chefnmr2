"""ChefNMR2's vote over a pooled candidate list, with the drop of implausible (charge-broken) candidates.

The paper's scoring code, with the charge rule of the inference API (`has_unpaired_charge`). `vote` gives
the ranked list, which needs no answer key; the rank lookup is separate. Its outputs are the paper's.

THE RULE
* Candidates are canonicalised under the matcher, then `Counter.most_common()` gives vote order;
  Python's stable sort makes ties keep first-seen order, so the order of the pooled list is
  load-bearing: callers pool members in ASCENDING EPOCH order, each member's draws in sampling order.
* A candidate with an unpaired formal charge on C, or a positive charge on O, is a bond-order
  reconstruction failure, not a hypothesis: the model has no charge channel. It is removed from the
  vote, keeping its votes in a separate list, unless every candidate is such a one.
* A rank beyond `list_len` scores as absent, matching a handed-in list of that length.

This is the paper's rule, and the inference API's (`chefnmr2.infer`): charge-broken candidates leave the
ranking, while their draws still count as valid.
"""

from __future__ import annotations

from collections import Counter

from chefnmr2.eval.scoring import canon


def has_unpaired_charge(smiles: str | None) -> bool:
    """True if this SMILES carries a formal charge that no stable compound has.

    A charge on carbon, or a POSITIVE charge on oxygen, is only ever real as one half of a
    charge-separated BOND -- isocyanide [C-]#[N+], ylides, oxocarbenium next to its anion.
    Unpaired, it is a bond-ORDER reconstruction failure: the model has no charge channel
    (atom_decoder is elements only), so every formal charge here is introduced downstream by
    bond reconstruction, which already tries to remove them. A charge that survives is one those
    fixes could not resolve.

    Charges on N / O- / S / P are NOT tested: zwitterions separate them by many bonds
    (betaine, amino acid, phosphate), and nitro / N-oxide / azide / sulfoxide are ordinary
    groups. Do NOT widen this to "any formal charge": that filter shipped once and was reverted;
    it rejects 43 of the 3,108 distinct molecules in the benchmark.

    Measured 2026-08-29: benchmark targets 0 of 4,047 rejected; the released model's training split 95 of
    2,524,941 (0.0038%: mesoionic sydnones, rhodamine / pyronin dyes, flavylium betaines); 0.97% of
    valid in-dist draws, 19.8% on an out-of-distribution set.
    """
    if not smiles:
        return False  # already a failed reconstruction; keep its own status
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False  # unparseable is a different failure, not this one
    for a in m.GetAtoms():
        q = a.GetFormalCharge()
        if q == 0:
            continue
        if (a.GetSymbol() == "C" or (a.GetSymbol() == "O" and q > 0)) and not any(
            n.GetFormalCharge() * q < 0 for n in a.GetNeighbors()
        ):
            return True
    return False


# The memo makes the drop one RDKit parse per distinct canonical candidate rather than one per draw.
# Canonicalization preserves formal charge, so that is the same verdict on every member of a vote group.
_IMPL: dict = {}


def _implausible(c):
    if c not in _IMPL:
        _IMPL[c] = has_unpaired_charge(c)
    return _IMPL[c]


def vote(pooled, remove_stereo, drop_implausible):
    """One matcher's vote over a pooled list -> (ranked, dropped).

    `ranked` is [(smiles, votes), ...] in vote order, `dropped` the implausible candidates removed from
    the vote, also in vote order. Failed reconstructions (empty or None) are not counted. Both are
    empty when no draw reconstructed.
    """
    canon_ = [canon(s, remove_stereo) if s else None for s in pooled]
    counts = Counter(c for c in canon_ if c)
    if not counts:
        return [], []

    bad = {c for c in counts if _implausible(c)} if drop_implausible else set()
    # Never empty the list. A fully charge-broken pool is a reconstruction failure the reader
    # must SEE -- scored, with its rank absent -- not an empty pool indistinguishable from a
    # molecule that was never sampled.
    if len(bad) == len(counts):
        bad = set()

    ranked = [(c, n) for c, n in counts.most_common() if c not in bad]
    dropped = [(c, n) for c, n in counts.most_common() if c in bad]
    return ranked, dropped


def pool_view(pooled, target, remove_stereo, list_len, drop_implausible):
    """One matcher's view of a pooled candidate list: the rank of the true structure.

    Returns (rank, ranked, dropped) as `vote` does, with `rank` the target's 1-based position in
    `ranked`. `rank` is None when the target is absent or sits past `list_len`. The target is
    compared with the candidates in `canon`'s form, so with `remove_stereo=True` this is
    `chefnmr2.chem.match.match_smiles`'s rule (same planar structure, stereochemistry ignored).

    Top-k accuracy over a set of molecules is the share whose `rank` is k or better.
    """
    ranked, dropped = vote(pooled, remove_stereo, drop_implausible)
    if not ranked and not dropped:
        return None, [], []
    tgt = canon(target, remove_stereo)
    rank = next((i + 1 for i, (c, _) in enumerate(ranked) if c == tgt), None)
    if rank is not None and list_len and rank > list_len:
        rank = None
    return rank, ranked, dropped

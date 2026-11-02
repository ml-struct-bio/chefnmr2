"""The canonical form candidates are compared in: `canon`, the matcher's canonicalization with a memo.

The matcher itself is not defined here: `canonicalize` is `chefnmr2.chem.smiles`'s. Reimplementing it
would be reimplementing the matcher, and a matcher that drifts is worth 12-26 pp while looking
completely normal.
"""

from __future__ import annotations

from rdkit import RDLogger

from chefnmr2.chem.smiles import canonicalize  # THE matcher

RDLogger.DisableLog("rdApp.*")

__all__ = ["canon", "canonicalize"]


# `canon` is a PURE function of its arguments, and a pooled list repeats the same structure many
# times over -- that repetition IS the vote. Without a memo, emitting B=1000 across 23 datasets
# is ~5.5M RDKit round-trips instead of ~0.3M. Semantically a no-op (verified byte-for-byte).
_CANON: dict = {}


def canon(s, remove_stereo):
    k = (s, remove_stereo)
    if k not in _CANON:
        _CANON[k] = canonicalize(s, remove_stereo=remove_stereo)
    return _CANON[k]

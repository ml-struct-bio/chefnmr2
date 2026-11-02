"""The paper's adaptive sampling: the opening call, the branch rule and the totals.

The rule opens every molecule with `OPENING_SAMPLES` draws, reads the pool for how much it agrees with itself
(`branch_of`), and spends `TOTAL[branch]` in total, the opening call included:

    converged   one candidate took nearly all the votes         100 draws
    divided     several candidates                              1,000 draws
    scattered   no confident candidates -- every attempt different, or none built anything   3,000 draws

`chefnmr2.infer` runs it: `elucidate(peaks, formula, n_samples="adaptive")`. Every total is drawn in equal
numbers from the `N_MEMBERS` checkpoints, and the pool is each checkpoint's draws in order (`prefix`).
"""

from __future__ import annotations

from chefnmr2.eval.vote import pool_view

# The opening call is a measurement, not a search, so its size is fixed: a vote fraction computed over
# a handful of attempts is trivially high, and would read as agreement.
OPENING_SAMPLES = 100

CONVERGED_VALIDITY = 0.10
CONVERGED_VOTE_FRACTION = 0.97
SCATTERED_UNIQUE_RATIO = 0.95

# Samples this molecule may spend in total, the opening call included.
TOTAL = {"converged": OPENING_SAMPLES, "divided": 1000, "scattered": 3000}

N_MEMBERS = 10


def branch_of(result: dict) -> tuple[str | None, dict]:
    """``(branch, evidence)`` for one response. ``branch`` is None when unreadable.

    First match wins. `evidence` carries the numbers the decision read together with the constants
    it read them against, so the branch stays re-derivable even after a constant here changes.

    A response missing any of those numbers returns no branch. Filing it under the widest one
    instead would hand out that branch's total on no evidence -- the reading has to be able to come
    back empty, or it is not a reading.
    """
    candidates = result.get("candidates") or []
    validity = result.get("validity")
    n_valid = result.get("n_valid")
    n_unique = result.get("n_unique_candidates")
    vote_fraction_0 = candidates[0].get("vote_fraction") if candidates else None

    evidence = {
        "validity": validity,
        "n_valid": n_valid,
        "n_unique_candidates": n_unique,
        "vote_fraction_0": vote_fraction_0,
        "thresholds": {
            "converged_validity": CONVERGED_VALIDITY,
            "converged_vote_fraction": CONVERGED_VOTE_FRACTION,
            "scattered_unique_ratio": SCATTERED_UNIQUE_RATIO,
        },
    }

    if validity is None or n_valid is None or n_unique is None:
        return None, evidence
    # Attempts built structures but no vote fraction came back with them: the converged test cannot
    # run, and the others would decide without it.
    if n_valid and vote_fraction_0 is None:
        return None, evidence

    if (
        validity >= CONVERGED_VALIDITY
        and vote_fraction_0 is not None
        and vote_fraction_0 >= CONVERGED_VOTE_FRACTION
    ):
        return "converged", evidence
    if n_valid == 0 or n_unique / n_valid >= SCATTERED_UNIQUE_RATIO:
        return "scattered", evidence
    return "divided", evidence


def prefix(members: list[list[str]], budget: int) -> list[str]:
    """The first `budget // N_MEMBERS` draws of each member, concatenated in member order."""
    take = max(1, budget // N_MEMBERS)
    return [s for m in members for s in m[:take]]


def response(pooled: list[str], target: str, list_len: int) -> dict:
    """Shape one pool like a serving response, so `branch_of` reads it unmodified.

    `vote_fraction` divides by the VOTED total -- implausible rows excluded -- which is the unit the
    branch thresholds expect.
    """
    _, ranked, dropped = pool_view(pooled, target, True, list_len, True)
    n_valid = sum(n for _, n in ranked) + sum(n for _, n in dropped)
    voted = sum(n for _, n in ranked)
    return {
        "validity": n_valid / len(pooled) if pooled else 0.0,
        "n_valid": n_valid,
        "n_unique_candidates": len(ranked) + len(dropped),
        "candidates": ([{"smiles": c, "vote_fraction": n / voted} for c, n in ranked[:1]] if voted else []),
    }

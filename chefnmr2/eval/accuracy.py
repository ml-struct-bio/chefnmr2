"""Top-k accuracy of ChefNMR2 on a test-set file, as the paper measures it.

    python -m chefnmr2.eval.accuracy data/specteach.jsonl [--checkpoints checkpoints] [--seed 0] [--out <jsonl>]

Builds `ChefNMR2` from the 10 checkpoint folders `<checkpoints>/chefnmr2-epoch*` (as the README's Quickstart
does), runs every row of the file with `elucidate(peaks, formula, n_samples="adaptive", seed=seed)`, the paper's
adaptive sampling, and prints top-1 to top-10 accuracy: the share of rows whose reported structure (`smiles`)
is among the first k candidates, a candidate being the reported structure when `match_smiles` says so.
`--out` writes one line per row: its id, the rank of the reported structure (None past 10), the samples drawn,
the branch of the adaptive sampling and the first 10 candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chefnmr2.chem.match import match_smiles

TOP_K = 10


def rank_of(row: dict, answer: dict) -> int | None:
    """The 1-based rank of the row's reported structure among the answer's first `TOP_K` candidates, or None."""
    for i, c in enumerate(answer["candidates"][:TOP_K]):
        if match_smiles(row["smiles"], c["smiles"]):
            return i + 1
    return None


def evaluate(model, rows: list[dict], seed: int = 0, on_row=None) -> list[dict]:
    """One result per row: `elucidate` with the adaptive sampling, and the rank of the reported structure."""
    from chefnmr2.infer import Peaks

    results = []
    for row in rows:
        peaks = Peaks(c=row["c"], h=[tuple(p) for p in row["h"]])
        answer = model.elucidate(peaks, formula=row["formula"], n_samples="adaptive", seed=seed)
        result = {
            "id": row["id"],
            "rank": rank_of(row, answer),
            "n_samples": answer["n_samples"],
            "branch": answer["adaptive"]["branch"],
            "candidates": [c["smiles"] for c in answer["candidates"][:TOP_K]],
        }
        results.append(result)
        if on_row is not None:
            on_row(len(results), len(rows), result)
    return results


def topk(results: list[dict]) -> list[tuple[int, int, float]]:
    """`(k, hits, share)` for k = 1 to `TOP_K`: the rows whose reported structure is among the first k."""
    n = len(results)
    out = []
    for k in range(1, TOP_K + 1):
        hits = sum(1 for r in results if r["rank"] is not None and r["rank"] <= k)
        out.append((k, hits, hits / n if n else 0.0))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="a test-set file, one JSON object per line (data/*.jsonl)")
    ap.add_argument("--checkpoints", default="checkpoints", help="the folder of the 10 checkpoint folders")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write one line per row to this JSON Lines file")
    a = ap.parse_args(argv)

    import torch

    from chefnmr2.infer import ChefNMR2

    rows = [json.loads(line) for line in Path(a.file).read_text().splitlines() if line.strip()]
    folders = sorted(Path(a.checkpoints).glob("chefnmr2-epoch*"))
    members = [(d.name, d / "model.pt", d / "config.yaml") for d in folders]
    model = ChefNMR2(members, device="cuda" if torch.cuda.is_available() else "cpu")

    def progress(i, n, r):
        print(f"{i}/{n} {r['id']}: rank {r['rank']}, {r['n_samples']} samples ({r['branch']})", flush=True)

    results = evaluate(model, rows, seed=a.seed, on_row=progress)
    if a.out:
        Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in results))
    print(f'{Path(a.file).name}: {len(results)} molecules, n_samples="adaptive", seed {a.seed}')
    for k, hits, share in topk(results):
        print(f"top-{k:<2} {100 * share:5.1f}%  ({hits}/{len(results)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

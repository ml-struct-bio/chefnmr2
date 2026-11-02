"""Judging ChefNMR2's predictions.

`vote`: the vote over a pooled candidate list, the charge rule (`has_unpaired_charge`) and the rank of the
true structure among the voted candidates (`pool_view`, from which top-k accuracy is counted). `scoring`:
the canonical form in which candidates are compared (`canon`). Whether a single prediction is correct is
`chefnmr2.chem.match.match_smiles`. `budget`: the paper's adaptive sampling (the opening call, the branch rule
and the totals), which `chefnmr2.infer` runs with `n_samples="adaptive"`. `accuracy`: top-k accuracy on a test-set
file, `python -m chefnmr2.eval.accuracy data/<file>.jsonl`.

Import the submodules directly; importing this package imports nothing else.
"""

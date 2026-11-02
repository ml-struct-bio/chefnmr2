"""Standalone ChefNMR2 inference: raw NMR peaks + molecular formula -> ranked candidate SMILES.

The chemistry comes from `chefnmr2.chem`, the charge filter is `chefnmr2.eval.vote.has_unpaired_charge`
(one copy of the rule), and the model is built by `chefnmr2.model.build`.

Consumes each checkpoint's two files: the EMA-folded weights (`model.pt`, ~1.15 GB fp32) and its
config (`config.yaml`).

Why the inputs can be sized to the request:
  - `model.model.sample()` (`AtomDiffusion.sample`) reads ONLY `model_inputs`.
  - The score model has no atom positional encoding and the peak embedder has no positional
    embedding, so the number of atoms and peaks is not tied to a fixed length.
  - EMA weights are folded into `model.pt`.

`ChefNMR2` runs the 10 released checkpoints and pools their samples into one vote-ranked
list, the configuration the paper's benchmark measures. See the class docstring for the pooling rule.
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
from omegaconf import OmegaConf

# Multiplet tokens as the data sets store them ...
_TOKEN = {"<pad>": 0, "s": 1, "d": 2, "t": 3, "q": 4, "p": 5, "h": 6, "hept": 7, "m": 8}
_TOKEN_RE = re.compile(r"(?:hept|s|d|t|q|p|h|m)")
# ... then collapsed to a single class.
# Anything not in this table (p, h, hept, ddd, ...) falls through to 8 == "m".
_CLASS = {
    (0, 0, 0): 0,
    (1, 0, 0): 1,
    (2, 0, 0): 2,
    (3, 0, 0): 3,
    (4, 0, 0): 4,
    (2, 2, 0): 5,
    (2, 3, 0): 6,
    (3, 2, 0): 7,
    (8, 0, 0): 8,
}

# The sampling settings of the paper, fixed for the released checkpoints.
_NUM_SAMPLING_STEPS = 50


def multiplet_class(mult: str | None) -> int:
    """'dd' -> 5, 'ddd' -> 8 ('m'), '' -> 0 (pad)."""
    if not mult:
        return 0
    toks = [_TOKEN[m.group(0)] for m in _TOKEN_RE.finditer(mult.strip().lower())]
    return _CLASS.get(tuple((toks + [0, 0, 0])[:3]), 8)


# A molecular formula and NOTHING else: element symbols with optional counts, end to end.
_FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")


def formula_to_counts(formula: str) -> dict[str, int]:
    """'C20H28O5' -> {'C': 20, 'H': 28, 'O': 5}. Must be the FULL formula including H.

    Rejects anything that is not a pure molecular formula, because the obvious alternative --
    a CONDENSED STRUCTURAL formula -- would otherwise parse silently into a different molecule. The
    element regex simply ignores characters it does not match, so parentheses and their
    multipliers would be dropped:

        "(CH3)2CHOH"   -> C2H5O   (isopropanol is C3H8O)   8 atoms instead of 12
        "CH3(CH2)4CH3" -> C3H8    (hexane is C6H14)

    Both pass every downstream check -- they contain H, use legal elements, and are under the
    atom cap -- so the call would return a confident elucidation of a molecule the caller never
    asked for, with no error anywhere.
    """
    if not _FORMULA_RE.match(formula.strip()):
        raise ValueError(
            f"{formula!r} is not a molecular formula. Give element symbols with counts and "
            f"nothing else (e.g. 'C3H8O'), not a condensed structural formula: '(CH3)2CHOH' "
            f"would silently parse as C2H5O and elucidate the wrong molecule."
        )
    out: dict[str, int] = {}
    for sym, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula.strip()):
        if sym:
            out[sym] = out.get(sym, 0) + (int(n) if n else 1)
    return out


# Per-sample reconstruction status, collapsed to three values.
#   value_error + valence collapse to `valence_error`: both are "no bond assignment satisfies
#   every atom's valence" reached by two RDKit paths (a ValueError from bond-order search vs an
#   AtomValenceException).
# Anything unmapped -> "other" (see .get default at the call site).
_STATUS = {
    "ok": "ok",  # connected molecule, SMILES produced
    "fragmented": "fragmented",  # bonds found but disconnected: smiles null, largest_fragment set
    "value_error": "valence_error",
    "valence": "valence_error",
}


# `_has_unpaired_charge` is `chefnmr2.eval.vote.has_unpaired_charge` itself: one copy of the rule for
# inference and the paper's scoring. A charge on carbon, or a positive charge on oxygen, with no neighbour of the opposite charge is a
# bond-order reconstruction failure; the measurements behind the rule are in that function's docstring.
from chefnmr2.eval.vote import vote as _paper_vote


def _rank(smiles_list) -> tuple[list[tuple[str, int]], list[tuple[str, int]], int]:
    """The paper's vote rule, `chefnmr2.eval.vote.vote`.

    Returns (ranked, implausible, n_voted). Draws are canonicalized stereo-blind and counted first;
    then the candidates whose canonical SMILES carries an unpaired formal charge
    (`has_unpaired_charge`) leave the ranking for `implausible`, unless every candidate is one.
    Both lists are in vote order; ties keep first appearance, so the order of the pooled list
    decides them. `n_voted` is the votes of the ranked candidates, the denominator of
    `vote_fraction`. The charge-broken draws stay reconstructed draws: they count in `n_valid` and
    `validity`.
    """
    ranked, implausible = _paper_vote(list(smiles_list), remove_stereo=True, drop_implausible=True)
    return ranked, implausible, sum(v for _, v in ranked)


# The adaptive sampling's rule (opening size, thresholds, totals, the per-member prefix) is the paper's,
# `chefnmr2.eval.budget`, the one copy its replay of the benchmark campaigns uses too.
from chefnmr2.eval import budget as _budget

# Draws per member in one call of the adaptive sampling after the opening call: 3,000 samples are drawn as
# calls of at most 100 per member, each at its own seeds, which bounds the GPU memory of one call.
_ADAPTIVE_CHUNK = 100


def _adaptive(draw, n_members: int, seed: int | None) -> tuple:
    """The paper's adaptive sampling over a sampler: `(branch, evidence, pooled)`.

    `draw(k, base_seed)` gives `k` new draws of each member, `[[member 0's], [member 1's], ...]`, each draw
    a tuple whose first item is its SMILES; member j samples at `base_seed + j + 1`. The opening call is
    `budget.OPENING_SAMPLES` draws at `seed`, read by `budget.branch_of` as the paper's replay reads it
    (`budget.response`). The rest of the branch's total is drawn in calls of at most `_ADAPTIVE_CHUNK` per
    member at base seeds `seed + n_members`, `seed + 2 * n_members`, ..., and each member's new draws extend
    its first ones, so the pool is the per-member prefix the replay takes of one long run (`budget.prefix`,
    member-major). With no branch (an unreadable opening) the opening call is the pool.
    """
    if n_members != _budget.N_MEMBERS:
        raise ValueError(f"the adaptive sampling runs all {_budget.N_MEMBERS} checkpoints, got {n_members}")
    members = [list(m) for m in draw(_budget.OPENING_SAMPLES // n_members, seed)]
    opening = [d[0] for d in _budget.prefix(members, _budget.OPENING_SAMPLES)]
    branch, evidence = _budget.branch_of(_budget.response(opening, None, 0))
    total = _budget.TOTAL[branch] if branch else _budget.OPENING_SAMPLES
    base = seed
    while len(members[0]) < total // n_members:
        base = None if base is None else base + n_members
        more = draw(min(_ADAPTIVE_CHUNK, total // n_members - len(members[0])), base)
        for m, d in zip(members, more):
            m.extend(d)
    return branch, evidence, _budget.prefix(members, total)


@dataclass
class Peaks:
    """One molecule's spectra. `h` entries are (shift_ppm, integration_nH, multiplet)."""

    c: list[float] = field(default_factory=list)
    h: list[tuple[float, float, str]] = field(default_factory=list)


def build_model_inputs(peaks: Peaks, formula: str, cfg, device: str = "cuda"):
    """Raw peaks + formula -> the `model_inputs` dict `AtomDiffusion.sample()` consumes.

    Padding convention throughout: real entries front-contiguous, mask 1.0 real / 0.0 pad.
    `atom_charges` is intentionally omitted — the model never reads it.
    """
    da = cfg.dataset_args
    decoder = list(da.atom_decoder)
    counts = formula_to_counts(formula)
    unknown = [s for s in counts if s not in decoder]
    if unknown:
        raise ValueError(f"formula has element(s) outside the atom decoder: {unknown}")

    # Size every tensor to THIS request, not to a fixed cap: nothing in the model is keyed to a
    # fixed length. The only real ceiling is memory (attention is O(n^2)).
    n_at = max(1, sum(counts.values()))
    n_h = max(1, len(peaks.h))
    n_c = max(1, len(peaks.c))

    h_cent = np.zeros(n_h, np.float32)
    h_nh = np.zeros(n_h, np.float32)
    h_mult = np.zeros(n_h, np.float32)
    h_mask = np.zeros(n_h, np.float32)
    for i, (ppm, nh, mult) in enumerate(peaks.h):
        h_cent[i], h_nh[i], h_mult[i], h_mask[i] = ppm, nh, multiplet_class(mult), 1.0

    c_cent = np.zeros(n_c, np.float32)
    c_mask = np.zeros(n_c, np.float32)
    for i, ppm in enumerate(peaks.c):
        c_cent[i], c_mask[i] = ppm, 1.0

    # The 13C list is used as given: close peaks are never merged, as in the released checkpoints'
    # training. Give one peak per distinct carbon environment, as experimental spectra report them.

    one_hot = np.zeros((n_at, len(decoder)), np.float32)
    atom_mask = np.zeros(n_at, np.float32)
    j = 0
    for sym, n in counts.items():
        col = decoder.index(sym)
        for _ in range(n):
            one_hot[j, col] = 1.0
            atom_mask[j] = 1.0
            j += 1

    to_t = lambda a: torch.from_numpy(a)[None].to(device)  # noqa: E731  (add batch dim)
    model_inputs = {
        "atom_mask": to_t(atom_mask),
        "atom_one_hot": to_t(one_hot),
        "condition": {
            "h_peak_centroid": to_t(h_cent),
            "h_peak_nH": to_t(h_nh),
            "h_peak_multiplet": to_t(h_mult),
            "h_peak_mask": to_t(h_mask),
            "c_peak_centroid": to_t(c_cent),
            "c_peak_mask": to_t(c_mask),
        },
    }
    return model_inputs, j


def _default_workers() -> int:
    # sched_getaffinity is Linux-only; fall back on macOS / Windows.
    return max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1))


class ReconPool:
    """A ProcessPoolExecutor that outlives the request, so per-member reconstruction can be
    submitted the moment a member finishes sampling and run on the CPUs while the GPU moves
    to the next member.

    Two reasons this exists rather than calling `chefnmr2.chem.bonds.atom_features_to_smiles` N times:

    * `atom_features_to_smiles` BLOCKS until every sample is done, which puts CPU reconstruction
      on the critical path between two GPU sampling runs. Submitting instead of blocking overlaps
      (N-1)/N of the reconstruction with sampling.
    * Workers are forked at construction, BEFORE any CUDA context exists.

    The worker function is the package's own `atom_features_to_smiles_one`, so reconstruction is
    bit-identical to the in-line path — this is a scheduling change, not a semantic one.
    """

    def __init__(self, num_workers: int | None = None):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        self.num_workers = num_workers or _default_workers()
        # "fork" explicitly, not the platform default: forking inherits the parent's
        # already-imported modules, which is why workers start instantly.
        self._pool = ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_recon_worker_init,
        )
        # Force ALL the forks now, before the caller puts anything on the GPU.
        # ProcessPoolExecutor spawns lazily and `_adjust_process_count` starts at most ONE
        # worker per submit, and only when no worker is idle — so a single warm-up task would
        # leave the other N-1 to fork mid-request, next to a live CUDA context. Submitting N
        # briefly-blocking tasks keeps every worker busy while the next submit lands, which is
        # what actually drives the count to max_workers.
        from concurrent.futures import wait

        wait([self._pool.submit(_recon_worker_warm) for _ in range(self.num_workers)])

    def submit(self, coords, one_hot, mask, da) -> list:
        """coords (K, N, 3); one_hot (N, D); mask (N,). Returns K futures, in sample order."""
        from chefnmr2.chem.bonds import atom_features_to_smiles_one

        return [
            self._pool.submit(
                atom_features_to_smiles_one,
                (
                    i,
                    coords[i],
                    one_hot,
                    mask,
                    bool(da.remove_h),
                    bool(da.remove_stereo),
                    list(da.atom_decoder),
                    da.max_iterations,
                ),
            )
            for i in range(len(coords))
        ]

    @staticmethod
    def drain(futures) -> tuple[list, list, list]:
        """(smiles, largest_fragment, raw_status) in submission order."""
        smiles, largest, status = [], [], []
        for f in futures:
            _i, smi, frag, err, _mol = f.result()
            smiles.append(smi)
            largest.append(frag)
            status.append(err)
        return smiles, largest, status

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


def _recon_worker_init():
    """Match `atom_features_to_smiles`'s log suppression. The in-line path calls this in the parent
    before forking; a persistent pool forks first, so the children have to do it themselves."""
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def _recon_worker_warm():
    """Occupy a worker just long enough that the next submit has to spawn another one.
    Module-level because a ProcessPoolExecutor task has to be picklable by reference."""
    time.sleep(0.05)


def _check_valid_equals_voted(smiles_list, n_voted: int) -> int:
    """Return n_valid, and warn if it is not equal to the vote total.

    `n_voted` is every counted vote, ranked and implausible. `vote_fraction` divides by the ranked
    votes (`_rank`), as the paper does, not by n_valid, so a gap here does not move the fractions,
    but it still means a reconstructed SMILES failed to canonicalize. The equality holds because
    every SMILES that reconstructed successfully has already survived a reparse
    (`CanonSmiles(MolToSmiles(mol))` in `chefnmr2.chem.bonds`), and the stereo strip of the vote's
    canonicalization was measured to keep every one of 38,200 reconstructed SMILES.

    So n_valid == sum(votes) is an argument about RDKit plus one measurement, not an invariant
    anything enforces. Check it rather than assume it: if RDKit ever breaks it, this line is how
    the log shows it.
    """
    n_valid = sum(1 for s in smiles_list if s)
    if n_voted != n_valid:
        print(
            f"WARNING: {n_valid - n_voted} sample(s) reconstructed but failed to canonicalize "
            f"(n_valid={n_valid}, sum(votes)={n_voted}). They are in n_valid but in no "
            f"candidate. This is expected to be impossible — see _check_valid_equals_voted."
        )
    return n_valid


def load_export(ckpt_path: str, cfg_path: str):
    """Read one checkpoint's weights and config -> (cfg, state_dict on CPU, epoch).

    `ckpt_path` is a released `model.pt`, `cfg_path` its `config.yaml`.
    """
    from chefnmr2.ckpt.format import load

    cfg = OmegaConf.load(cfg_path)
    obj = load(ckpt_path, "inference", mmap=False)
    export = obj["export"]
    if export.get("ema_applied") is not True:
        raise RuntimeError(
            f"{ckpt_path} does not hold EMA-folded weights; the released checkpoints all do."
        )
    state = dict(obj["weights"])
    epoch = export.get("epoch")
    del obj
    return cfg, state, epoch


class _Checkpoint:
    """One checkpoint, built once: its config, its model on `device`, its epoch and tag."""

    def __init__(self, ckpt_path: str, cfg_path: str, device: str = "cuda", tag: str | None = None):
        from chefnmr2.model.build import build_model

        cfg, state, epoch = load_export(ckpt_path, cfg_path)
        self.cfg = cfg
        model = build_model(cfg)
        model.load_state_dict(state, strict=True)
        del state
        self.model = model.eval().to(device)
        self.device = device
        self.tag = tag
        self.epoch = epoch
        self.default_guidance = float(self.model.model.guidance_scale)

    @torch.no_grad()
    def sample(self, model_inputs: dict, n: int):
        """`n` diffusion samples for one molecule -> coords (n, N, 3), on the model's device.

        `n` is a BATCH multiplier, not a loop counter: all `n` candidates are denoised together in
        one pass, and classifier-free guidance doubles that batch again inside the score model.
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            coords = self.model.model.sample(
                model_inputs=model_inputs,
                num_sampling_steps=_NUM_SAMPLING_STEPS,
                multiplicity=n,
            )
        return coords


# Keys whose value changes how coordinates are DECODED into SMILES. Two ensemble members that
# disagree on any of them would decode the shared pool under mixed rules, with no error anywhere.
_RECON_KEYS = ("atom_decoder", "remove_h", "remove_stereo", "max_iterations")
# Keys that are legitimately per-checkpoint, so members are NOT required to agree — each keeps
# its own cfg — but where disagreement almost certainly means a mis-assembled member list.
#   sigma_data     the coordinate scale, per training set.
#   guidance_scale each member samples under ITS OWN guidance, while params.guidance_scale
#                  reports member 0's.
_WARN_KEYS = ("sigma_data", "guidance_scale")


class ChefNMR2:
    """The 10 released checkpoints, pooled into a single vote-ranked list.

    This is the configuration the paper's benchmark measures: no checkpoint is ever selected; every
    member samples K candidates from the same spectra and the union is vote-ranked.

    The paper's pooling rule:

      1. members in ASCENDING EPOCH order,
      2. each contributes exactly K candidates in raw sampling order, K applied per member
         BEFORE pooling — this is what makes the weighting equal,
      3. concatenate, canonicalize, drop failures, `Counter().most_common()`.
         `most_common()` is a stable sort, so TIES GO TO THE EARLIEST EPOCH.
      4. no per-member dedup and no per-member weighting; candidates with an unpaired formal
         charge leave the ranking for `implausible_candidates` (`chefnmr2.eval.vote.vote`).

    Candidates are compared stereo-blind (planar structure only). `vote_fraction` divides a
    candidate's votes by the votes of all ranked candidates (the samples that produced a parseable
    structure), as the paper's sampling frequency does.

    Per-member sampling seed is `seed + member_index + 1` (checkpoint j gets seed j + 1 at the
    default `seed=0`), as in the paper's runs.
    """

    def __init__(self, members: list, device: str = "cuda", recon_workers: int | None = None):
        """`members` is [(tag, model_pt, config_yaml), ...] in any order — sorted here by epoch.

        `device` is "cuda" or "cpu". `recon_workers=None` reconstructs structures in a pool of one
        worker per CPU, built before any CUDA context; `recon_workers=0` reconstructs in the calling
        process, for hosted environments that do not allow a persistent pool.
        """
        if not members:
            raise ValueError("ChefNMR2 needs at least one member")

        self.recon_pool = None
        if recon_workers != 0:
            self.recon_pool = ReconPool(recon_workers)

        self.device = device

        recs = []
        for tag, ckpt_path, cfg_path in members:
            chef = _Checkpoint(ckpt_path=ckpt_path, cfg_path=cfg_path, device=device, tag=tag)
            recs.append({"tag": tag, "epoch": chef.epoch, "cfg": chef.cfg, "chef": chef})

        # Ascending epoch is load-bearing for tie-breaking, not cosmetic (see the class
        # docstring). A member with no epoch recorded sorts last but keeps a stable order.
        recs.sort(
            key=lambda r: (r["epoch"] is None, r["epoch"] if r["epoch"] is not None else 0, r["tag"] or "")
        )
        self.members = recs
        self._check_member_agreement()

        self.cfg = recs[0]["cfg"]  # reconstruction + input-building conventions
        self.default_guidance = recs[0]["chef"].default_guidance

        self.tags = [r["tag"] for r in recs]
        self.epochs = [r["epoch"] for r in recs]

    def _check_member_agreement(self):
        ref = self.members[0]
        for m in self.members[1:]:
            bad = {
                k: (ref["cfg"].dataset_args.get(k), m["cfg"].dataset_args.get(k))
                for k in _RECON_KEYS
                if str(ref["cfg"].dataset_args.get(k)) != str(m["cfg"].dataset_args.get(k))
            }
            if bad:
                raise RuntimeError(
                    f"ensemble members {ref['tag']!r} and {m['tag']!r} disagree on "
                    f"reconstruction conventions, which would decode one pooled candidate list "
                    f"under two sets of rules with no error:\n"
                    + "\n".join(f"  dataset_args.{k}: {a!r} vs {b!r}" for k, (a, b) in bad.items())
                )
            for k in _WARN_KEYS:
                a = ref["cfg"].diffusion_process_args.get(k)
                b = m["cfg"].diffusion_process_args.get(k)
                if a != b:
                    print(
                        f"WARNING: {ref['tag']} has {k}={a} but {m['tag']} has {k}={b}. That "
                        f"is legal (it is per-training-set) but two members of one ensemble "
                        f"should normally come from one run — check the member list."
                    )

    @torch.no_grad()
    def _draw(
        self, chosen: list, model_inputs: dict, one_hot, mask, da, k: int, budget: int, seed: int | None
    ) -> tuple:
        """`k` samples from each member of `chosen`, reconstructed into SMILES.

        Returns `(smiles, largest, status, per_member, coord_blocks, sample_s, recon_s)`: the per-draw
        lists in pooled order (member 0's `k` draws, then member 1's, ...), one record per member, the
        coordinate blocks, and the seconds spent sampling and waiting for reconstruction. Member `idx`
        samples at `seed + idx + 1`; `budget` is `k * len(chosen)`.
        """
        from chefnmr2.chem import bonds as bond_analyzer

        sample_s = 0.0
        per_member, coord_blocks, futures = [], [], []
        for idx, m in chosen:
            # Per-member reseeding is what makes `seed` mean anything here. Without it a
            # member's noise depends on how much RNG the PREVIOUS member consumed.
            if seed is not None:
                torch.manual_seed(seed + idx + 1)
            t0 = time.time()
            coords = m["chef"].sample(model_inputs, k)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            sample_s += time.time() - t0

            arr = coords.float().cpu().numpy()
            coord_blocks.append(arr)
            per_member.append(
                {
                    "tag": m["tag"],
                    "epoch": m["epoch"],
                    "index": idx,
                    "seed": None if seed is None else seed + idx + 1,
                    "n_samples": k,
                }
            )
            # Submit, do not block: these K reconstructions run on the CPUs while the NEXT
            # member is sampling on the GPU.
            if self.recon_pool is not None:
                futures.append(self.recon_pool.submit(arr, one_hot, mask, da))

        # NB `recon_s` below is the time spent WAITING for reconstruction, not the CPU time it
        # took: with the persistent pool most of it already happened while the GPU was on the
        # next member. `timing_s.total` is the number to compare across configurations.
        t1 = time.time()
        drained = None
        if self.recon_pool is not None:
            try:
                smiles, largest, status = [], [], []
                for f in futures:
                    s, l, e = ReconPool.drain(f)
                    smiles += s
                    largest += l
                    status += e
                drained = True
            except Exception as e:  # BrokenProcessPool and friends
                # A persistent pool that breaks stays broken, which would wedge every later
                # call. Drop it permanently and reconstruct in-line (a fresh pool per call)
                # instead; not rebuilt, since a new fork would now happen next to a live CUDA
                # context. The process degrades in speed, not in correctness.
                print(
                    f"WARNING: reconstruction pool failed ({type(e).__name__}: {e}); "
                    f"this process has dropped to the in-line reconstruction path"
                )
                try:
                    self.recon_pool.shutdown()
                except Exception:
                    pass
                self.recon_pool = None
                drained = False
        if not drained:
            smiles, largest, _c, status = bond_analyzer.atom_features_to_smiles(
                {
                    "atom_coords": np.concatenate(coord_blocks, axis=0),
                    "atom_one_hot": model_inputs["atom_one_hot"].cpu().numpy(),
                    "atom_mask": model_inputs["atom_mask"].cpu().numpy(),
                },
                budget,
                remove_h=da.remove_h,
                remove_stereo=da.remove_stereo,
                atom_decoder=list(da.atom_decoder),
                max_iterations=da.max_iterations,
                num_workers=_default_workers(),
            )
        recon_s = time.time() - t1
        return smiles, largest, status, per_member, coord_blocks, sample_s, recon_s

    @torch.no_grad()
    def elucidate(self, peaks: Peaks, formula: str, n_samples: int | str = 100, seed: int | None = 0) -> dict:
        """Sample, reconstruct and vote-rank candidate structures for one molecule.

        `n_samples` is the TOTAL budget across the ensemble, not per checkpoint. It is rounded UP
        to a multiple of the member count so every member contributes the same K; the returned
        `n_samples` is the budget actually spent. `seed` makes the call reproducible on one
        machine (`None` draws fresh noise).

        Every member always runs, with the paper's settings: 50 sampling steps and each
        checkpoint's own guidance scale.

        `n_samples="adaptive"` samples as the paper does (`_adaptive`): 100 samples, then 1,000 or 3,000
        in total by what the first 100 gave. The answer adds `adaptive` (the branch and the opening call's
        numbers), and `n_samples` is the samples drawn.
        """
        if isinstance(n_samples, str) and n_samples == "adaptive":
            return self._elucidate_adaptive(peaks, formula, seed)
        chosen = list(enumerate(self.members))
        n_members = len(chosen)
        k = max(1, -(-int(n_samples) // n_members))  # ceil, so never fewer than requested
        budget = k * n_members

        model_inputs, _n_atoms = build_model_inputs(peaks, formula, self.cfg, self.device)
        da = self.cfg.dataset_args
        one_hot = model_inputs["atom_one_hot"][0].cpu().numpy()
        mask = model_inputs["atom_mask"][0].cpu().numpy()

        wall0 = time.time()
        smiles, largest, status, per_member, coord_blocks, sample_s, recon_s = self._draw(
            chosen, model_inputs, one_hot, mask, da, k, budget, seed
        )
        return self._answer(
            smiles=smiles,
            status=status,
            per_member=per_member,
            sample_s=sample_s,
            recon_s=recon_s,
            wall0=wall0,
            k=k,
            budget=budget,
            n_members=n_members,
            seed=seed,
        )

    @torch.no_grad()
    def _elucidate_adaptive(self, peaks: Peaks, formula: str, seed: int | None) -> dict:
        """`elucidate` with `n_samples="adaptive"`: the draws of `_adaptive`, answered by `_answer`."""
        chosen = list(enumerate(self.members))
        n_members = len(chosen)

        model_inputs, _n_atoms = build_model_inputs(peaks, formula, self.cfg, self.device)
        da = self.cfg.dataset_args
        one_hot = model_inputs["atom_one_hot"][0].cpu().numpy()
        mask = model_inputs["atom_mask"][0].cpu().numpy()

        wall0 = time.time()
        calls = []

        def draw(k, base):
            out = self._draw(chosen, model_inputs, one_hot, mask, da, k, k * n_members, base)
            calls.append(out)
            return [list(zip(*(x[j * k : (j + 1) * k] for x in out[:3]))) for j in range(n_members)]

        branch, evidence, pooled = _adaptive(draw, n_members, seed)
        budget = len(pooled)
        k = budget // n_members
        smiles, largest, status = (list(x) for x in zip(*pooled))
        # One record per member, over all its calls, as one long run would give.
        per_member = [
            dict(m, n_samples=k, seeds=[c[3][j]["seed"] for c in calls]) for j, m in enumerate(calls[0][3])
        ]
        out = self._answer(
            smiles=smiles,
            status=status,
            per_member=per_member,
            sample_s=sum(c[5] for c in calls),
            recon_s=sum(c[6] for c in calls),
            wall0=wall0,
            k=k,
            budget=budget,
            n_members=n_members,
            seed=seed,
        )
        keys = ("validity", "n_valid", "n_unique_candidates", "vote_fraction_0")
        out["adaptive"] = {
            "branch": branch,
            "opening": {"n_samples": _budget.OPENING_SAMPLES, **{key: evidence[key] for key in keys}},
            "thresholds": evidence["thresholds"],
            "totals": dict(_budget.TOTAL),
        }
        return out

    def _answer(
        self,
        smiles,
        status,
        per_member,
        sample_s,
        recon_s,
        wall0,
        k,
        budget,
        n_members,
        seed,
    ) -> dict:
        """The answer of one call: its draws in pooled order, as `_draw` returns them, and its settings."""
        # The paper's rule: the per-member `n_valid` and `validity` below count charge-broken draws
        # as reconstructed, as the paper does.
        ranked, implausible, n_voted = _rank(smiles)

        for j, m in enumerate(per_member):
            lo = j * k
            m["n_valid"] = sum(1 for s in smiles[lo : lo + k] if s)
            m["validity"] = round(m["n_valid"] / k, 4) if k else 0.0

        # Denominator is sum(votes) of the ranked candidates -- the samples that produced a
        # parseable structure, less the implausible candidates' votes -- NOT the budget, as in
        # the paper: this says how concentrated the vote is GIVEN a structure was built, and
        # `n_valid` separately says how often one could be built at all.
        n_valid = _check_valid_equals_voted(smiles, n_voted + sum(v for _, v in implausible))
        return {
            "candidates": [
                {"smiles": s, "votes": v, "vote_fraction": round(v / n_voted, 4) if n_voted else 0.0}
                for s, v in ranked
            ],
            # The candidates the paper's rule takes out of the ranking, in vote order: an unpaired
            # formal charge on C, or a positive one on O, is a bond-order reconstruction failure.
            # Empty when none, or when every candidate is one (then they rank).
            "implausible_candidates": [{"smiles": s, "votes": v} for s, v in implausible],
            # The three counts of the ladder, in order. Each is a different quantity and none is
            # `len` of another: n_samples DRAWS -> n_valid that built a structure -> the survivors
            # merged by vote into n_unique_candidates distinct answers, ranked or implausible.
            "n_samples": budget,
            "n_valid": n_valid,
            "n_unique_candidates": len(ranked) + len(implausible),
            # n_valid / n_samples, the form that is comparable across budgets.
            "validity": round(n_valid / budget, 4) if budget else 0.0,
            # How the budget split across reconstruction outcomes. Sums to n_samples, and
            # status_counts["ok"] == n_valid.
            "status_counts": dict(Counter(_STATUS.get(s, "other") for s in status).most_common()),
            "timing_s": {
                "sample": round(sample_s, 2),
                "reconstruct": round(recon_s, 2),
                "total": round(time.time() - wall0, 2),
            },
            "ensemble": {
                "n_checkpoints": n_members,
                "n_available": len(self.members),
                "n_samples_per_checkpoint": k,
                "members": per_member,
            },
            "params": {
                "num_sampling_steps": _NUM_SAMPLING_STEPS,
                "guidance_scale": self.default_guidance,
                "seed": seed,
                "return_raw_samples": False,
                # Bumped when the meaning of n_valid / validity / vote_fraction changes. 3 = the
                # paper's rule: charge-broken draws count in n_valid, their candidates leave the
                # ranking.
                "validity_definition": 3,
                "checkpoint_epoch": [m["epoch"] for m in per_member],
            },
        }

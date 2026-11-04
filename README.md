<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/chefnmr2-logo-dark.svg">
    <img src="assets/chefnmr2-logo.svg" alt="ChefNMR2" width="160">
  </picture>
</p>

[Preprint](https://doi.org/10.64898/2026.10.29.762711) · [Project page](https://chefnmr.cs.princeton.edu) · [Web app](https://huggingface.co/spaces/ziyux123/chefnmr2) · [Weights](https://huggingface.co/ziyux123/chefnmr2)

ChefNMR2 elucidates the structure of a small molecule from its 1H and 13C NMR peak lists and molecular formula.

<p align="center"><img src="assets/chefnmr2-fig1ab.png" alt="Structure elucidation from 1D NMR spectra, and approaches to it" width="100%"></p>

This repository holds ChefNMR2's inference code; the training code and training data follow with the journal publication.

## How to use

- **ChefNMR2**
  - **Chat app**: connect ChefNMR2 to an AI chat app such as Claude or ChatGPT and ask for a structure in plain words, as [Use in an AI chat app](#use-in-an-ai-chat-app-mcp) shows.
  - **Web demo**: run ChefNMR2 in the browser with the [Web app](https://huggingface.co/spaces/ziyux123/chefnmr2), with no install.
  - **Install and run locally**: install ChefNMR2, download the paper's checkpoints from Hugging Face and elucidate a structure, as the sections below show.
- **ChefNMR2-Agent**: coming soon.

## Use in an AI chat app (MCP)

ChefNMR2 works in a chat app through MCP, the Model Context Protocol, an open standard that lets an AI chat app use outside tools. Once ChefNMR2 is connected, you paste your NMR data in plain words, the chat app sends the formula and peak lists to ChefNMR2, and it shows you the candidate structures.

What you need: a Claude or ChatGPT Plus account and a free Hugging Face account. ChefNMR2 runs on Hugging Face's shared GPUs (ZeroGPU), and each call uses the daily GPU time of the Hugging Face account you sign in with.

1. Add ChefNMR2 to your chat app, with the address `https://huggingface.co/mcp?login&bouquet=no_gradio_images&gradio=ziyux123/chefnmr2`.
   - **Claude** (claude.ai or the desktop app): open **Customize > Connectors**, click **+ Add** and then **Add custom connector**. Name it `chefnmr2`, paste the address, choose **Sign in now** and click **Add**. On a Team or Enterprise plan, ask an Owner to add it for the whole organization; each member then connects it and signs in.
   - **ChatGPT or Codex**: open **Plugins** and add a custom MCP server named `chefnmr2` with the address, then connect it.
2. Sign in to Hugging Face on the page that opens, or make a free account there.
3. Ask for a structure. Paste the formula and peak lists in any format, for example:

   > Here are the NMR data of an unknown compound, C17H19NO3. 13C NMR: δ 146.4, 138.6, 133.5, 128.8, 127.4, 124.9, 119.5, 116.7, 91.4, 66.4, 58.8, 46.8, 43.0, 42.7, 40.7, 35.7, 20.5. 1H NMR: δ 6.72 (d, 1H), 6.55 (d, 1H), 5.70 (d, 1H), 5.26 (d, 1H), 4.86 (d, 1H), 4.16 (m, 1H), 3.32 (m, 1H), 2.98 (d, 1H), 2.85 (m, 1H), 2.57 (dd, 1H), 2.42 (s, 3H), 2.28 (td, 1H), 2.05 (m, 1H), 1.86 (m, 1H), 1.60 (m, 1H). Use ChefNMR2 to elucidate its structure.

   The answer shows the formula and peaks ChefNMR2 received, so you can check them against your data, and the candidate structures as SMILES, ranked by sampling frequency. ChefNMR2's sampling frequency is calibrated, so you can read it as the probability that the structure is correct. ChefNMR2 runs 100 samples first. When the top structure's sampling frequency is below 0.97, or more than nine in ten samples give no structure, the chat app says that the answer is not highly confident and offers to run 1000 samples, which uses more of your daily GPU time. For this example the top structure is morphine (planar), at about 0.92 with 100 samples and about 0.93 with 1000. The paper's results use an automatic rule instead: 100 samples, then 1,000 or 3,000 in total under the same two conditions (`n_samples="adaptive"` in the Quickstart below).

Limits:

- Hugging Face's daily GPU time is 5 minutes with a free account and 40 minutes with PRO. When it is used up, the call fails with a message that says so; the time comes back 24 hours after its first use.
- A molecule above about 110 atoms, hydrogens included (morphine has 40), may need more GPU time than a free account has in a day. Then use a PRO account, or install ChefNMR2 and run it on your own GPU, as the sections below show.
- When a call fails, the chat app should tell you so. A structure it gives without ChefNMR2's answer is not ChefNMR2's.
- The web app does not store or log your inputs; they are processed on Hugging Face's servers. What you paste into the chat app follows that app's own terms.

## Install

Python 3.11 on Linux.

```bash
git clone https://github.com/ml-struct-bio/chefnmr2
cd chefnmr2
pip install .
python -c "from chefnmr2.infer import ChefNMR2, Peaks; print('ok')"
```

`pip install .` takes PyTorch 2.5.1 from PyPI (with CUDA 12 on Linux). For a CPU-only machine, install the CPU wheel first: `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu`.

## Checkpoints

The 10 checkpoints of the paper's model (one training run, epochs 949 to 1399 every 50) are on Hugging Face, in the model repository `ziyux123/chefnmr2`. Each folder holds `model.pt` (fp32, 1.15 GB), `config.yaml` and `manifest.json` (the sha256 of `model.pt` and `config.yaml`). Download them (11.5 GB) into `checkpoints/`; this and the commands below run from the repository's folder:

<!-- not run: a download; the checks place these files themselves -->
```bash
pip install huggingface_hub
hf download ziyux123/chefnmr2 --revision 38a741cce3e6f7e0d5a827a66f8d3dea0728daf5 --local-dir checkpoints
```

Then check every file against its folder's `manifest.json`:

```python
import hashlib
import json
from pathlib import Path

folders = sorted(p for p in Path("checkpoints").iterdir() if (p / "manifest.json").is_file())
bad = []
for d in folders:
    for name, f in json.loads((d / "manifest.json").read_text())["files"].items():
        h = hashlib.sha256()
        with open(d / name, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != f["sha256"]:
            bad.append(f"{d.name}/{name}")
print(f"{len(folders)} checkpoints, sha256 checked: {'all match' if not bad else bad}")
```

```text
10 checkpoints, sha256 checked: all match
```

## Quickstart

Elucidate one molecule and read the answer. Elucidate morphine from its peak lists:

```python
import json
from pathlib import Path

import torch
from chefnmr2.infer import ChefNMR2, Peaks

members = [(d.name, d / "model.pt", d / "config.yaml") for d in sorted(Path("checkpoints").glob("chefnmr2-epoch*"))]
model = ChefNMR2(members, device="cuda" if torch.cuda.is_available() else "cpu")

peaks = Peaks(
    c=[145.2, 138.0, 133.2, 130.9, 128.5, 126.6, 120.2, 116.8, 91.9,
       66.6, 59.1, 46.6, 43.3, 43.1, 40.7, 35.6, 20.7],
    h=[(6.63, 1, "d"), (6.5, 1, "d"), (5.67, 1, "dddd"), (5.28, 1, "ddd"),
       (4.89, 1, "dd"), (4.2, 1, "ddd"), (3.41, 1, "dd"), (3.03, 1, "d"),
       (2.73, 1, "m"), (2.66, 1, "dd"), (2.47, 3, "s"), (2.45, 1, "td"),
       (2.33, 1, "dd"), (2.12, 1, "td"), (1.88, 1, "ddd")],
)
out = model.elucidate(peaks, formula="C17H19NO3", n_samples=100, seed=0)
print(json.dumps(out["candidates"], indent=1))
```

Expected output (on an NVIDIA H100 GPU):

<!-- output: gpu only -->
```text
[
 {
  "smiles": "CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5",
  "votes": 94,
  "vote_fraction": 0.9895
 },
 {
  "smiles": "CN1CCC23C4=CCC(O)C2Oc2c(O)ccc(c23)CC41",
  "votes": 1,
  "vote_fraction": 0.0105
 }
]
```

- Input:
  - `peaks`: the 13C shifts in ppm, and the 1H peaks as (shift in ppm, number of protons, multiplicity).
  - `formula`: the molecular formula, with hydrogens.
  - `n_samples`: how many samples to draw.
    - A number, such as `n_samples=100` above: the samples are split evenly over the checkpoints (10 each here).
    - `"adaptive"`: sample as the paper does: 100 samples first, then 1,000 or 3,000 in total when they do not agree on one structure (Methods); the answer's `n_samples` gives the number drawn.
  - `seed`: the random seed.
- Output: `candidates`, the structures the samples gave, ranked by sampling frequency.
  - `smiles`: the structure.
  - `votes`: the number of samples that gave it.
  - `vote_fraction`: its sampling frequency.

Accuracy metric: the paper reports top-k accuracy, the share of molecules whose reported structure is among the first k candidates; `match_smiles` decides whether a candidate is the reported structure (the same planar structure, stereochemistry ignored).

<!-- continues: the Quickstart -->
```python
from chefnmr2.chem.match import match_smiles

reported = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"  # morphine, as reported
hits = [match_smiles(reported, c["smiles"]) for c in out["candidates"]]
print({f"top-{k}": any(hits[:k]) for k in (1, 3, 10)})
```

```text
{'top-1': True, 'top-3': True, 'top-10': True}
```

Time for this call (PyTorch 2.5.1):

| hardware | load the 10 checkpoints | `elucidate` (100 samples) |
|---|--:|--:|
| NVIDIA H100 80GB HBM3 (CUDA 12.4) | 14.6 to 15.1 s | 5.7 to 6.9 s |
| NVIDIA A100-SXM4-80GB (CUDA 12.4) | 16.1 s | 9.7 to 11.0 s |
| 16 CPU threads (Intel Xeon Platinum 8462Y+ or AMD EPYC 7413) | 14.4 to 17.2 s | 74.4 to 178.3 s |

## Test sets

Run ChefNMR2 on the paper's test sets: `data/` holds them as used in the study, one file per set, with each molecule's formula, peak lists and reported structure.

- [data/README.md](data/README.md) lists the files, their fields, their sources and the licence of each file.
- A row's `c` and `h` go into `Peaks` as they are: `Peaks(c=row["c"], h=[tuple(p) for p in row["h"]])`.
- Reproduce ChefNMR2's results on one file: the command below runs every row with `n_samples="adaptive"` and prints top-1 to top-10 accuracy by `match_smiles`. On one NVIDIA A100 GPU it takes about 45 min for `specteach.jsonl` and about 3 h for `nmrshiftdb2.jsonl`.

<!-- not run: hours on a GPU -->
```bash
python -m chefnmr2.eval.accuracy data/specteach.jsonl --checkpoints checkpoints
```

## Citation

Cite the preprint (also in [CITATION.cff](CITATION.cff)):

```bibtex
@article{xiong2026chefnmr2,
  title   = {Small molecule structure elucidation from NMR spectra with generative and reasoning models},
  author  = {Xiong, Ziyu and Ying, Vanessa Y. and Huang, Jonathan Z. and Koch, Zane and Clark, Joseph D. and Heeter, Robert C. and An, Joon Soo and Matanachai, Karit and Rodriques, Samuel G. and White, Andrew D. and Seyedsayamdost, Mohammad R. and Zhong, Ellen D.},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.10.29.762711}
}
```

## Licence

- The code is under the MIT License ([LICENSE](LICENSE)); third-party code and its licences are in [THIRD_PARTY.md](THIRD_PARTY.md).
- The checkpoints are under the Community Data License Agreement – Sharing 1.0 (CDLA-Sharing-1.0).
- The checkpoints were trained on data that include structures from NPAtlas and NP-MRD (CC BY-NC 4.0); check those terms before any commercial use.

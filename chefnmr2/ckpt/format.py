"""The file format of the released checkpoints (`model.pt`).

A checkpoint is a `torch.save` of a dict that holds only tensors and plain Python values, so `load`
reads it with `torch.load(weights_only=True)`: reading it runs no pickled code. The released
checkpoints are of the `inference` kind: the EMA-folded weights, epoch, step and the checkpoint's
name; `load` refuses a file without this format's marker.
"""

import torch

FORMAT = "chefnmr2-checkpoint"
FORMAT_VERSION = 1


class FormatError(ValueError):
    """A file that is not a checkpoint of the expected kind and version."""


def load(path, kind: str, mmap: bool = True) -> dict:
    """Read a checkpoint of `kind` on the CPU (memory-mapped unless `mmap=False`), refusing anything else."""
    if kind != "inference":
        raise ValueError(f"kind must be 'inference', got {kind!r}")
    obj = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    if not isinstance(obj, dict) or obj.get("format") != FORMAT:
        raise FormatError(f"{path} is not a {FORMAT} file")
    if obj.get("format_version") != FORMAT_VERSION:
        raise FormatError(
            f"{path}: format_version {obj.get('format_version')!r}, this code reads {FORMAT_VERSION}"
        )
    if obj.get("kind") != kind:
        raise FormatError(f"{path} is a {obj.get('kind')!r} checkpoint, not {kind!r}")
    return obj

"""ChefNMR2 inference: NMR peaks and a molecular formula to ranked candidate structures.

    from chefnmr2.infer import ChefNMR2, Peaks

`ChefNMR2` runs the 10 released checkpoints together, as the paper's benchmark does. The code is in
`api.py`.
"""

from chefnmr2.infer.api import ChefNMR2, Peaks

__all__ = ["ChefNMR2", "Peaks"]

"""OOD voxel evaluation on fine_grid subset (samples 0-999, one inclusion)."""
from pathlib import Path

from eval.ood.eval_ood_voxel import REPO_ROOT, run_ood_eval

OUTPUT_ROOT = str(REPO_ROOT / "eval_outputs" / "ood_fine_grid")

if __name__ == "__main__":
    run_ood_eval(subset="fine_grid", default_output_root=OUTPUT_ROOT)

"""OOD voxel evaluation on two_inclusions subset (samples 1000-1999, two inclusions)."""
from pathlib import Path

from eval.ood.eval_ood_voxel import REPO_ROOT, run_ood_eval

OUTPUT_ROOT = str(REPO_ROOT / "eval_outputs" / "ood_two_inclusions")

if __name__ == "__main__":
    run_ood_eval(subset="two_inclusions", default_output_root=OUTPUT_ROOT)

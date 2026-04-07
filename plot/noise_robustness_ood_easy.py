"""Plot OOD noise robustness curves and generate LaTeX tables for the easy dataset.

Covers fine_grid and two_inclusions subsets for the no-encoder models.

Usage:
    PYTHONPATH=. uv run python plot/noise_robustness_ood_easy.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Serif"

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "eval_outputs"

SWEEP_PATHS = {
    "fine_grid": OUTPUT_DIR / "ood_fine_grid_easy-noise" / "noise_sweep.json",
    "two_inclusions": OUTPUT_DIR / "ood_two_inclusions_easy-noise" / "noise_sweep.json",
}

NOISE_LEVELS = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

METRICS = ["mean_rel_l2", "mean_rel_l1"]
METRIC_LABELS = {
    "mean_rel_l2": r"Rel-$\ell_2$",
    "mean_rel_l1": r"Rel-$\ell_1$",
}

SUBSET_LABELS = {
    "fine_grid": "Fine Grid (OOD)",
    "two_inclusions": "Two Inclusions (OOD)",
}

_CYCLE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f"]


def _find_best_at(sweep: dict, group: dict, mk: str) -> dict[float, str]:
    best_at: dict[float, str] = {}
    for nl in NOISE_LEVELS:
        best_run, best_val = None, float("inf")
        for run in group:
            for e in sweep[run]:
                if e["noise_level"] == nl and e[mk] is not None and e[mk] < best_val:
                    best_val = e[mk]
                    best_run = run
        best_at[nl] = best_run
    return best_at


def _find_top_models(sweep: dict) -> dict[str, str]:
    """Find models that are ever best at any noise level for any metric."""
    ever_best = set()
    for mk in METRICS:
        ba = _find_best_at(sweep, sweep, mk)
        ever_best.update(ba.values())
    ever_best.discard(None)

    def sort_key(m):
        for e in sweep[m]:
            if e["noise_level"] == 0.0 and e["mean_rel_l2"] is not None:
                return e["mean_rel_l2"]
        return 999

    ordered = sorted(ever_best, key=sort_key)

    labels = {}
    for m in ordered:
        parts = m.split("_")
        lr = [p for p in parts if p.startswith("lr")][0] if any(p.startswith("lr") for p in parts) else ""
        bs = [p for p in parts if p.startswith("bs")][0] if any(p.startswith("bs") for p in parts) else ""
        ch = [p for p in parts if p.startswith("ch")][0] if any(p.startswith("ch") for p in parts) else ""
        wd = [p for p in parts if p.startswith("wd")][0] if any(p.startswith("wd") for p in parts) else ""
        labels[m] = f"{lr} {bs} {ch} {wd}".strip()
    return {m: labels[m] for m in ordered}


def plot_easy(sweep: dict, top_models: dict[str, str], subset: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(f"Noise Robustness — Easy Dataset — {SUBSET_LABELS[subset]}", fontsize=14, fontweight="bold")

    for ax_i, (mk, mk_label) in enumerate(METRIC_LABELS.items()):
        ax = axes[ax_i]
        best_at = _find_best_at(sweep, top_models, mk)

        best_at_zero = best_at.get(0.0)

        for i, (run, label) in enumerate(top_models.items()):
            color = "#1f77b4" if run == best_at_zero else _CYCLE[(i % (len(_CYCLE) - 1)) + 1]
            data = {e["noise_level"]: e[mk] for e in sweep[run] if e[mk] is not None}
            xs = [nl for nl in NOISE_LEVELS if nl in data]
            ys = [data[nl] for nl in xs]

            star_xs = [nl for nl in xs if best_at.get(nl) == run]
            star_ys = [data[nl] for nl in star_xs]

            alpha = 1.0 if run == best_at_zero else 0.35
            ax.plot(xs, ys, color=color, linewidth=1.5, label=label, alpha=alpha)
            if star_xs:
                ax.scatter(star_xs, star_ys, marker="*", s=80, c=color,
                           zorder=5, edgecolors="black", linewidths=0.4, alpha=alpha)

        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_yscale("log")
        ax.set_xlabel("Noise level", fontsize=11)
        ax.set_ylabel(mk_label, fontsize=11)
        ax.set_title(mk_label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(NOISE_LEVELS)
        ax.set_xticklabels([str(n) for n in NOISE_LEVELS], rotation=45, ha="right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTPUT_DIR / f"noise_robustness_ood_{subset}_easy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _fmt(v: float) -> str:
    if v is None:
        return "--"
    if abs(v) >= 100:
        return f"{v:.1f}"
    elif abs(v) >= 1:
        return f"{v:.3f}"
    else:
        return f"{v:.4f}"


def latex_tables(sweep: dict, top_models: dict[str, str], subset: str) -> str:
    lines: list[str] = []

    best_at: dict[tuple[str, float], str] = {}
    for mk in METRICS:
        ba = _find_best_at(sweep, top_models, mk)
        for nl, run in ba.items():
            best_at[(mk, nl)] = run

    n = len(top_models)
    lines.append(r"\begin{figure}[H]")
    lines.append(r"    \centering")

    for i, (model, label) in enumerate(top_models.items()):
        data = {e["noise_level"]: e for e in sweep[model]}

        col_spec = "r rr"
        header = r"\textbf{Noise} & \textbf{Rel-$\ell_2$} & \textbf{Rel-$\ell_1$} \\"

        lines.append(r"    \begin{subfigure}[t]{\dimexpr0.32\textwidth}")
        lines.append(r"        \centering")
        lines.append(r"        \small")
        lines.append(f"        \\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"        \toprule")
        lines.append(f"        {header}")
        lines.append(r"        \midrule")

        for nl in NOISE_LEVELS:
            e = data.get(nl, {})
            cells = []
            for mk in METRICS:
                v = e.get(mk)
                s = _fmt(v)
                if best_at.get((mk, nl)) == model:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            lines.append(f"        {nl} & {' & '.join(cells)} \\\\")

        lines.append(r"        \bottomrule")
        lines.append(r"        \end{tabular}")
        lines.append(f"        \\caption{{{label}.}}")
        lines.append(r"    \end{subfigure}")
        if i < n - 1:
            lines.append(r"    \hfill")

    safe_subset = subset.replace("_", " ")
    lines.append(f"    \\caption{{Noise robustness, easy dataset, {safe_subset} (OOD). Best value per noise level in \\textbf{{bold}}.}}")
    lines.append(f"    \\label{{fig:noise-tables-ood-{subset}-easy}}")
    lines.append(r"\end{figure}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    for subset, sweep_path in SWEEP_PATHS.items():
        if not sweep_path.exists():
            print(f"Skipping {subset}: {sweep_path} not found")
            continue

        sweep = json.loads(sweep_path.read_text())
        top_models = _find_top_models(sweep)

        print(f"\n=== {SUBSET_LABELS[subset]} ===")
        print(f"Top models ({len(top_models)}):")
        for m, l in top_models.items():
            print(f"  {l}: {m}")

        plot_easy(sweep, top_models, subset)

        tex = latex_tables(sweep, top_models, subset)
        out = OUTPUT_DIR / f"noise_robustness_ood_{subset}_easy_tables.tex"
        out.write_text(tex)
        print(f"Saved {out}")
        print(tex)


if __name__ == "__main__":
    main()

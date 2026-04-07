"""Plot noise robustness curves and generate LaTeX tables from noise_sweep.json.

Usage:
    PYTHONPATH=. uv run python plot/noise_robustness.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Serif"

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = REPO_ROOT / "eval_outputs" / "grid_search_2-unnormalized-noise" / "noise_sweep.json"
OUTPUT_DIR = REPO_ROOT / "eval_outputs"

NOISE_LEVELS = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

SPARSE_MODELS = {
    "_sparse_pixel_only_lr5e-04_bs64_ep800_unet_ch32_pixel": "Physical-only",
    "_sparse_antipodal_lr5e-04_bs64_ep800_unet_ch32_pixel": "Both (physical output)",
    "_sparse_antipodal_lr5e-04_bs128_ep800_unet_ch32_latent": "Both (latent output)",
    "_sparse_latent_only_lr5e-04_bs128_ep800_unet_ch32_latent": "Latent-only",
}

DENSE_MODELS = {
    "_dense_pixel_only_lr5e-04_bs32_ep800_unet_ch32_pixel": "Physical-only",
    "_dense_antipodal_lr5e-04_bs32_ep800_unet_ch32_pixel": "Both (physical output)",
    "_dense_antipodal_lr5e-04_bs64_ep800_unet_ch32_latent": "Both (latent output)",
    "_dense_latent_only_lr5e-04_bs64_ep800_unet_ch32_latent": "Latent-only",
}

COLORS = {
    "Physical-only": "#d62728",
    "Both (physical output)": "#ff7f0e",
    "Both (latent output)": "#1f77b4",
    "Latent-only": "#9467bd",
}

METRICS = ["mean_bs_rel_l2", "mean_bs_rel_l1"]
METRIC_LABELS = {
    "mean_bs_rel_l2": r"BS-Rel-$\ell_2$",
    "mean_bs_rel_l1": r"BS-Rel-$\ell_1$",
}
METRIC_LATEX = {
    "mean_bs_rel_l2": r"BS-Rel-$\ell_2$",
    "mean_bs_rel_l1": r"BS-Rel-$\ell_1$",
}


# ── plotting ──────────────────────────────────────────────────────────────────

def _find_best_at(sweep: dict, group: dict, mk: str) -> dict[float, str]:
    best_at: dict[float, str] = {}
    for nl in NOISE_LEVELS:
        best_run, best_val = None, float("inf")
        for run in group:
            for e in sweep[run]:
                if e["noise_level"] == nl and e[mk] < best_val:
                    best_val = e[mk]
                    best_run = run
        best_at[nl] = best_run
    return best_at


def plot_group(sweep: dict, group_name: str, group: dict[str, str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(f"Noise Robustness — {group_name} Dataset", fontsize=14, fontweight="bold")

    for ax_i, (mk, mk_label) in enumerate(METRIC_LABELS.items()):
        ax = axes[ax_i]
        best_at = _find_best_at(sweep, group, mk)

        for run, label in group.items():
            data = {e["noise_level"]: e[mk] for e in sweep[run]}
            xs = NOISE_LEVELS
            ys = [data[nl] for nl in xs]

            star_xs = [nl for nl in xs if best_at[nl] == run]
            star_ys = [data[nl] for nl in star_xs]

            ax.plot(xs, ys, color=COLORS[label], linewidth=1.5, label=label)

            if star_xs:
                ax.scatter(
                    star_xs, star_ys, marker="*", s=80, c=COLORS[label],
                    zorder=5, edgecolors="black", linewidths=0.4,
                )

        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_yscale("log")
        ax.set_xlabel("Noise level", fontsize=11)
        ax.set_ylabel(mk_label, fontsize=11)
        ax.set_title(mk_label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(NOISE_LEVELS)
        ax.set_xticklabels([str(n) for n in NOISE_LEVELS], rotation=45, ha="right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTPUT_DIR / f"noise_robustness_{group_name.lower()}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_combined(sweep: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)

    for row_i, (group_name, group) in enumerate([("Sparse", SPARSE_MODELS), ("Dense", DENSE_MODELS)]):
        for col_i, (mk, mk_label) in enumerate(METRIC_LABELS.items()):
            ax = axes[row_i, col_i]
            best_at = _find_best_at(sweep, group, mk)

            for run, label in group.items():
                data = {e["noise_level"]: e[mk] for e in sweep[run]}
                xs = NOISE_LEVELS
                ys = [data[nl] for nl in xs]

                star_xs = [nl for nl in xs if best_at[nl] == run]
                star_ys = [data[nl] for nl in star_xs]

                ax.plot(xs, ys, color=COLORS[label], linewidth=1.5, label=label)

                if star_xs:
                    ax.scatter(
                        star_xs, star_ys, marker="*", s=80, c=COLORS[label],
                        zorder=5, edgecolors="black", linewidths=0.4,
                    )

            ax.set_xscale("symlog", linthresh=0.001)
            ax.set_yscale("log")
            ax.set_xlabel("Noise level", fontsize=10)
            ax.set_ylabel(mk_label, fontsize=10)
            ax.set_title(f"{group_name} — {mk_label}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, alpha=0.3)
            ax.set_xticks(NOISE_LEVELS)
            ax.set_xticklabels([str(n) for n in NOISE_LEVELS], rotation=45, ha="right", fontsize=7)

    fig.tight_layout()
    out = OUTPUT_DIR / "noise_robustness_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── LaTeX tables ──────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.1f}"
    elif abs(v) >= 1:
        return f"{v:.3f}"
    else:
        return f"{v:.4f}"


def latex_tables(sweep: dict, group_name: str, group: dict[str, str]) -> str:
    lines: list[str] = []

    # find best-in-class per (metric, noise_level)
    best_at: dict[tuple[str, float], str] = {}
    for mk in METRICS:
        ba = _find_best_at(sweep, group, mk)
        for nl, run in ba.items():
            best_at[(mk, nl)] = run

    # include all models, sorted by zero-noise bs_rel_l2
    def sort_key(m):
        for e in sweep[m]:
            if e["noise_level"] == 0.0:
                return e["mean_bs_rel_l2"]
        return 999

    ordered = sorted(group.keys(), key=sort_key)

    n = len(ordered)

    lines.append(r"\begin{figure}[H]")
    lines.append(r"    \centering")

    # first subfigure gets the Noise column, rest don't
    for i, model in enumerate(ordered):
        label = group[model]
        data = {e["noise_level"]: e for e in sweep[model]}
        is_first = (i == 0)

        if is_first:
            col_spec = "r rr"
            header = r"\textbf{Noise} & \textbf{BS-Rel-$\ell_2$} & \textbf{BS-Rel-$\ell_1$} \\"
        else:
            col_spec = "rr"
            header = r"\textbf{BS-Rel-$\ell_2$} & \textbf{BS-Rel-$\ell_1$} \\"

        lines.append(r"    \begin{subfigure}[t]{\dimexpr0.24\textwidth}")
        lines.append(r"        \centering")
        lines.append(r"        \small")
        lines.append(f"        \\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"        \toprule")
        lines.append(f"        {header}")
        lines.append(r"        \midrule")

        for nl in NOISE_LEVELS:
            e = data[nl]
            cells = []
            for mk in METRICS:
                v = e[mk]
                s = _fmt(v)
                if best_at[(mk, nl)] == model:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            if is_first:
                lines.append(f"        {nl} & {' & '.join(cells)} \\\\")
            else:
                lines.append(f"        {' & '.join(cells)} \\\\")

        lines.append(r"        \bottomrule")
        lines.append(r"        \end{tabular}")
        lines.append(f"        \\caption{{{label}.}}")
        lines.append(r"    \end{subfigure}")
        if i < n - 1:
            lines.append(r"    \hfill")

    safe = group_name.lower()
    lines.append(f"    \\caption{{Noise robustness, {safe} dataset. Best value per noise level in \\textbf{{bold}}.}}")
    lines.append(f"    \\label{{fig:noise-tables-{safe}}}")
    lines.append(r"\end{figure}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    sweep = json.loads(SWEEP_PATH.read_text())

    # individual plots
    plot_group(sweep, "Sparse", SPARSE_MODELS)
    plot_group(sweep, "Dense", DENSE_MODELS)

    # combined stacked figure
    plot_combined(sweep)

    # tables
    tex = ""
    tex += latex_tables(sweep, "Sparse", SPARSE_MODELS)
    tex += latex_tables(sweep, "Dense", DENSE_MODELS)

    out = OUTPUT_DIR / "noise_robustness_tables.tex"
    out.write_text(tex)
    print(f"Saved {out}")
    print(tex)


if __name__ == "__main__":
    main()

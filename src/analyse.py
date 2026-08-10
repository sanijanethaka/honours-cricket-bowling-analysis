"""
Cross-clip and cross-model analysis.

    python src/analyse.py

Reads Outputs/runs/<clip>/<backend>/angles.csv and produces:

  Outputs/model_agreement.csv   per-clip agreement between the two estimators
  Outputs/figures/agreement_<clip>.png   overlay + Bland-Altman per clip
  Outputs/figures/signals_by_style.png   all clips' signals, normalised time
  Outputs/figures/features_by_style.png  feature comparison, fast vs spin

The agreement analysis is the semester's central experiment: with the pipeline
held constant and only the estimator swapped, any disagreement is attributable
to the pose model. Disagreement is reported against the ICC 15 degree tolerance
as a reference scale for whether the choice of estimator is material at the
precision this application needs.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np              # noqa: E402
import pandas as pd             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import angles as A  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "Outputs" / "runs"
FIGS = ROOT / "Outputs" / "figures"

FOREST, MOSS, RED, GREY = "#2C5F2D", "#97BC62", "#B03A2E", "#5A6B5B"


def available_runs() -> dict[str, dict[str, pd.DataFrame]]:
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for angles_csv in sorted(RUNS.glob("*/*/angles.csv")):
        clip, backend = angles_csv.parent.parent.name, angles_csv.parent.name
        out.setdefault(clip, {})[backend] = pd.read_csv(angles_csv)
    return out


def agreement_table(runs) -> pd.DataFrame:
    rows = []
    for clip, by_backend in runs.items():
        for b1, b2 in combinations(sorted(by_backend), 2):
            rows.append({
                "clip_id": clip, "backend_a": b1, "backend_b": b2,
                **A.agreement(by_backend[b1], by_backend[b2]),
            })
    return pd.DataFrame(rows)


def plot_agreement(clip: str, by_backend: dict[str, pd.DataFrame]) -> None:
    names = sorted(by_backend)
    if len(names) < 2:
        return
    a, b = by_backend[names[0]], by_backend[names[1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    for df, name, col in ((a, names[0], FOREST), (b, names[1], RED)):
        ax1.plot(df["frame"], df["elbow_angle_smoothed"], lw=2, color=col, label=name)
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Elbow angle (degrees)")
    ax1.set_title(f"{clip} — same pipeline, two estimators")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    j = pd.DataFrame({
        "a": a.set_index("frame")["elbow_angle_filtered"],
        "b": b.set_index("frame")["elbow_angle_filtered"],
    }).dropna()
    if not j.empty:
        mean, diff = (j.a + j.b) / 2, j.a - j.b
        bias, sd = diff.mean(), diff.std(ddof=1) if len(j) > 1 else 0.0
        ax2.scatter(mean, diff, s=14, color=FOREST, alpha=0.6)
        ax2.axhline(bias, color=FOREST, lw=1.6, label=f"bias {bias:+.1f}°")
        for k, lbl in ((1.96, "95% LoA"), (-1.96, None)):
            ax2.axhline(bias + k * sd, color=GREY, ls="--", lw=1.1, label=lbl)
        for k in (1, -1):
            ax2.axhline(k * A.ICC_TOLERANCE_DEG, color=RED, ls=":", lw=1.3,
                        label="ICC ±15° tolerance" if k == 1 else None)
        ax2.set_xlabel("Mean of the two estimators (degrees)")
        ax2.set_ylabel(f"{names[0]} − {names[1]} (degrees)")
        ax2.set_title(f"Bland–Altman, n={len(j)} shared frames")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8, loc="best")
    else:
        ax2.text(0.5, 0.5, "no shared frames survived filtering",
                 ha="center", va="center", color=RED)
        ax2.set_axis_off()

    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"agreement_{clip}.png", dpi=150)
    plt.close(fig)


def plot_signals_by_style(features: pd.DataFrame, runs, backend: str) -> None:
    """
    All clips on one axis, time normalised to the retained delivery window.

    Normalised time is what makes a 30 fps clip and a 60 fps slow-motion clip
    comparable at all; plotting raw frame indices across mixed capture rates
    compares durations, not motions.
    """
    meta = features[features["backend"] == backend].set_index("clip_id")
    fig, ax = plt.subplots(figsize=(11, 5.4))
    styles = {"fast": FOREST, "spin": RED}
    seen = set()
    for clip, by_backend in sorted(runs.items()):
        if backend not in by_backend or clip not in meta.index:
            continue
        row = meta.loc[clip]
        if str(row.get("status")) != "included":
            continue
        # Keep NaN gaps rather than dropping them: concatenating retained
        # frames would silently close the windows the visibility filter opened
        # and make a fragmented signal look continuous.
        sig = by_backend[backend]
        s = sig["elbow_angle_smoothed"]
        if s.notna().sum() == 0:
            continue
        t = sig["frame"].values / max(sig["frame"].max(), 1) * 100
        cat = str(row.get("category", "fast"))
        lbl = cat if cat not in seen else None
        seen.add(cat)
        ax.plot(t, s.values, lw=2, alpha=0.85,
                color=styles.get(cat, GREY), label=lbl)
        last = s.last_valid_index()
        ax.annotate(f"{row.get('bowler', clip)}  ({row['retention']:.0%} retained)",
                    (t[s.index.get_loc(last)], s.loc[last]),
                    fontsize=8, color=GREY, xytext=(4, 0),
                    textcoords="offset points", va="center")
    ax.axhline(180, color=GREY, ls=":", lw=1, label="full extension (180°)")
    ax.set_xlabel("Normalised delivery progress (% of retained frames)")
    ax.set_ylabel("Elbow angle (degrees)")
    ax.set_title(f"Delivery-phase elbow signals by bowling style — {backend}")
    ax.set_xlim(0, 135)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "signals_by_style.png", dpi=150)
    plt.close(fig)


def plot_features_by_style(features: pd.DataFrame) -> None:
    df = features[(features["status"] == "included") & features["peak_extension_deg"].notna()]
    if df.empty:
        return
    metrics = [
        ("peak_extension_deg", "Peak extension (°)"),
        ("angular_range_deg", "Angular range (°)"),
        ("max_rate_deg_per_s", "Max rate of change (°/s)"),
        ("retention", "Keypoint retention"),
    ]
    backends = sorted(df["backend"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.4))
    palette = [FOREST, RED, MOSS]
    for ax, (col, label) in zip(np.atleast_1d(axes), metrics):
        clips = sorted(df["clip_id"].unique())
        x = np.arange(len(clips))
        w = 0.8 / max(len(backends), 1)
        for i, backend in enumerate(backends):
            sub = df[df["backend"] == backend].set_index("clip_id")
            vals = [sub[col].get(c, np.nan) for c in clips]
            ax.bar(x + i * w, vals, w, color=palette[i % len(palette)],
                   label=backend if col == metrics[0][0] else None)
        ax.set_xticks(x + w * (len(backends) - 1) / 2)
        ax.set_xticklabels(clips, rotation=30, ha="right", fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.3, axis="y")
        if col == "peak_extension_deg":
            ax.axhline(180, color=GREY, ls=":", lw=1)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(backends),
                   fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "features_by_style.png", dpi=150)
    plt.close(fig)


def main() -> int:
    runs = available_runs()
    if not runs:
        raise SystemExit(f"no runs found under {RUNS}; run src/pipeline.py first")

    features = pd.read_csv(ROOT / "Outputs" / "features.csv")

    table = agreement_table(runs)
    if not table.empty:
        out = ROOT / "Outputs" / "model_agreement.csv"
        table.to_csv(out, index=False)
        print(f"Wrote {out.relative_to(ROOT)}")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    for clip, by_backend in runs.items():
        plot_agreement(clip, by_backend)
    backend = "mediapipe" if any("mediapipe" in v for v in runs.values()) else \
        sorted(next(iter(runs.values())))[0]
    plot_signals_by_style(features, runs, backend)
    plot_features_by_style(features)
    print(f"Figures written to {FIGS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

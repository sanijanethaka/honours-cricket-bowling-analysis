"""
Rubric-critical experiments: statistical validity, occlusion decomposition,
resolution sensitivity.

    python src/experiments.py stats
    python src/experiments.py occlusion
    python src/experiments.py resolution --limit 20

Why this module exists
----------------------
The thesis marking rubric awards 20 of 70 points to a criterion whose
"Excellent" band requires "statistically valid results and conclusions" and the
ability to "generalise outside the immediate project tasks". Reporting a mean
absolute error of 19.7 deg satisfies neither on its own:

  * A point estimate with no interval is not a statistically valid result.
  * ~11,000 frames drawn from 160 sequences are not 11,000 independent
    observations. Frames within a sequence share an athlete, a camera, a
    lighting condition and a pose-tracker state. Treating them as independent
    inflates the effective sample size by roughly two orders of magnitude and
    makes every p-value meaningless.
  * A single lumped error figure cannot say what the error is MADE OF, so it
    cannot say how much of it is reducible.

Each subcommand below addresses one of those.

    stats       sequence-clustered confidence intervals, a paired
                backend comparison, and a test of the viewpoint effect
    occlusion   decomposes error against annotated body visibility
    resolution  decomposes error against image resolution

`stats` and `occlusion` reuse the CSVs already written by src/penn_action.py --
no model inference, so they run in seconds. `resolution` re-runs inference and
is the slow one.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import angles as A                              # noqa: E402
import penn_action as P                          # noqa: E402
from backends import get_backend                 # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Outputs" / "penn_action"
EXP = ROOT / "Outputs" / "experiments"

RNG_SEED = 20260824          # fixed so every reported interval is reproducible
N_BOOT = 10_000

FOREST, RED, GREY = "#2C5F2D", "#B03A2E", "#5A6B5B"


# --------------------------------------------------------------------------- #
# Shared statistics helpers
# --------------------------------------------------------------------------- #

def boot_ci(values: np.ndarray, stat=np.mean, n_boot: int = N_BOOT,
            alpha: float = 0.05, seed: int = RNG_SEED) -> tuple[float, float, float]:
    """
    Percentile bootstrap CI for a statistic of one sample.

    The resampling unit is whatever row you pass in. Pass SEQUENCE-level values,
    never frame-level ones: resampling frames would treat correlated
    observations as independent, which is exactly the error this module exists
    to avoid.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    if v.size == 1:
        return float(stat(v)), np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    dist = stat(v[idx], axis=1)
    lo, hi = np.percentile(dist, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(v)), float(lo), float(hi)


def fmt_ci(point: float, lo: float, hi: float, unit: str = "°") -> str:
    if not np.isfinite(point):
        return "n/a"
    if not np.isfinite(lo):
        return f"{point:.1f}{unit}"
    return f"{point:.1f}{unit} [{lo:.1f}, {hi:.1f}]"


def _require(path: Path, how: str, dtype=None) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(
            f"missing {path.relative_to(ROOT)}\n"
            f"Run this first:  {how}"
        )
    return pd.read_csv(path, dtype=dtype)


def seq_key(value) -> str:
    """
    Canonical form of a Penn Action sequence id.

    The ids are zero-padded strings ('0001'), and a CSV round-trip through
    pandas silently reads them as integers, so '0001' from the .mat filename
    never matches 1 from the CSV. Normalising both sides through this function
    is the fix. Stripping the padding rather than restoring it also survives the
    reverse case, where a file was written unpadded.
    """
    s = str(value).strip()
    stripped = s.lstrip("0")
    return stripped if stripped else "0"


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #

def cmd_stats(args) -> int:
    """
    Sequence-clustered inference on the existing validation run.

    Produces three things the thesis needs:

    1. Headline error per backend, with a 95% CI, bootstrapped over SEQUENCES.
    2. A PAIRED backend comparison. Both backends ran on the same sequences, so
       the correct comparison is within-sequence differences, not two
       independent means. Pairing removes between-sequence variance, which is
       the dominant variance component here.
    3. A test of the viewpoint effect, which is Chapter 3's planar-geometry
       prediction. Reporting front 15.5 vs back 23.2 without a test states a
       difference; testing it establishes one.
    """
    df = _require(OUT / "sequence_scores.csv",
                  "python src/penn_action.py --limit 40")
    df = df.dropna(subset=["mae_deg"])
    if df.empty:
        raise SystemExit("sequence_scores.csv has no scored sequences")

    EXP.mkdir(parents=True, exist_ok=True)
    rows = []

    # ---- 1. headline error per backend, clustered by sequence ---------------
    print("\n=== Headline error, 95% CI bootstrapped over sequences ===")
    print("Resampling unit is the sequence, not the frame. n is the number of")
    print("sequences, which is the study's true sample size.\n")
    for b, sub in df.groupby("backend"):
        for metric in ("mae_deg", "rmse_deg", "bias_deg", "frac_within_icc"):
            pt, lo, hi = boot_ci(sub[metric].values)
            unit = "" if metric == "frac_within_icc" else "°"
            rows.append({"analysis": "headline", "backend": b, "metric": metric,
                         "n_sequences": len(sub), "point": pt, "ci_lo": lo,
                         "ci_hi": hi})
            if metric in ("mae_deg", "frac_within_icc"):
                label = "MAE" if metric == "mae_deg" else "within ICC ±15°"
                print(f"  {b:<12} n={len(sub):<4} {label:<16} "
                      f"{fmt_ci(pt, lo, hi, unit)}")

    # ---- 2. paired backend comparison --------------------------------------
    backends = sorted(df["backend"].unique())
    if len(backends) >= 2:
        a, b = backends[0], backends[1]
        wide = (df.pivot_table(index="sequence_id", columns="backend",
                               values="mae_deg")
                  .dropna(subset=[a, b]))
        if len(wide) >= 3:
            d = (wide[a] - wide[b]).values
            pt, lo, hi = boot_ci(d)
            print(f"\n=== Paired comparison: {a} minus {b} ===")
            print(f"  {len(wide)} sequences scored by both backends")
            print(f"  mean paired difference {fmt_ci(pt, lo, hi)}")
            try:
                from scipy.stats import wilcoxon
                stat, pval = wilcoxon(wide[a], wide[b])
                print(f"  Wilcoxon signed-rank W={stat:.0f}, p={pval:.4f}")
            except Exception as exc:                     # pragma: no cover
                pval = np.nan
                print(f"  (Wilcoxon unavailable: {exc})")
            # The two tests answer different questions and can disagree. The
            # bootstrap CI is on the MEAN difference, which a handful of
            # badly-failing sequences can destabilise. Wilcoxon is rank-based:
            # it asks whether one estimator wins in a consistent DIRECTION,
            # regardless of magnitude. Reporting only whichever one suits the
            # narrative would be selective reporting, so print both and say
            # what the combination means.
            crosses_zero = np.isfinite(lo) and lo <= 0 <= hi
            sig = np.isfinite(pval) and pval < 0.05
            better = b if pt > 0 else a
            print("  interpretation:")
            if crosses_zero and sig:
                print(f"    direction is consistent ({better} wins in more "
                      "sequences than not, per Wilcoxon), but the mean")
                print("    difference is small and imprecisely estimated. Report "
                      "this as a statistically")
                print("    detectable but practically negligible difference — it "
                      "is an order of magnitude")
                print(f"    smaller than the {A.ICC_TOLERANCE_DEG:.0f}° tolerance "
                      "that matters. Do not pick a 'winner'.")
            elif crosses_zero:
                print("    the CI includes zero and Wilcoxon is not significant: "
                      "these two estimators")
                print("    are not distinguishable on this sample.")
            else:
                print(f"    both tests agree: {better} is measurably more "
                      "accurate on this sample.")
            rows.append({"analysis": "paired_backend",
                         "backend": f"{a}-{b}", "metric": "mae_diff_deg",
                         "n_sequences": len(wide), "point": pt,
                         "ci_lo": lo, "ci_hi": hi, "p_value": pval})

    # ---- 3. viewpoint effect ------------------------------------------------
    print("\n=== Viewpoint effect (Chapter 3's planar-geometry prediction) ===")
    for b, sub in df.groupby("backend"):
        groups = [g["mae_deg"].values for _, g in sub.groupby("viewpoint")
                  if len(g) >= 2]
        names = [v for v, g in sub.groupby("viewpoint") if len(g) >= 2]
        if len(groups) < 2:
            continue
        try:
            from scipy.stats import kruskal
            stat, pval = kruskal(*groups)
            print(f"  {b:<12} Kruskal–Wallis H={stat:.2f}, p={pval:.4f} "
                  f"across {len(groups)} viewpoints")
        except Exception as exc:                         # pragma: no cover
            stat = pval = np.nan
            print(f"  {b:<12} (Kruskal–Wallis unavailable: {exc})")
        rows.append({"analysis": "viewpoint_kruskal", "backend": b,
                     "metric": "mae_deg", "n_sequences": len(sub),
                     "point": stat, "p_value": pval})

        # The specific contrast the thesis claims: front versus back.
        if {"front", "back"} <= set(names):
            f = sub[sub["viewpoint"] == "front"]["mae_deg"].values
            k = sub[sub["viewpoint"] == "back"]["mae_deg"].values
            diff = np.mean(k) - np.mean(f)
            rng = np.random.default_rng(RNG_SEED)
            dist = (rng.choice(k, (N_BOOT, k.size)).mean(axis=1)
                    - rng.choice(f, (N_BOOT, f.size)).mean(axis=1))
            lo, hi = np.percentile(dist, [2.5, 97.5])
            print(f"               back minus front {fmt_ci(diff, lo, hi)} "
                  f"(n_front={f.size}, n_back={k.size})")
            rows.append({"analysis": "viewpoint_back_minus_front",
                         "backend": b, "metric": "mae_diff_deg",
                         "n_sequences": f.size + k.size, "point": diff,
                         "ci_lo": float(lo), "ci_hi": float(hi)})
            # An omnibus test that fails while this contrast succeeds is not a
            # contradiction, and the pattern is informative rather than
            # awkward. Obliquity predicts a MONOTONIC rise as the camera
            # rotates away from side-on, which would show up as an omnibus
            # effect. Occlusion predicts that only the view where the torso
            # hides the arm differs. A null omnibus with a real front-vs-back
            # gap is the occlusion signature, not the obliquity one.
            if np.isfinite(pval) and pval >= 0.05 and np.isfinite(lo) and lo > 0:
                print("               note: omnibus null but this contrast "
                      "non-null. That pattern is what")
                print("                     occlusion predicts and what "
                      "obliquity does not — obliquity")
                print("                     implies a monotonic rise across all "
                      "four viewpoints. Confirm with")
                print("                     `python src/experiments.py occlusion`"
                      " before relying on it.")

    # ---- optional: frame-level mixed model, for completeness ---------------
    pf = OUT / "per_frame.csv"
    if pf.is_file() and not args.no_mixed:
        _mixed_model(pd.read_csv(pf), rows)

    out = EXP / "stats_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out.relative_to(ROOT)}")
    print("\nFor the thesis: quote the CI alongside every headline number, and")
    print("state explicitly that the resampling unit is the sequence. That")
    print("sentence is what makes the analysis defensible to a second marker.")
    return 0


def _mixed_model(frames: pd.DataFrame, rows: list) -> None:
    """
    Frame-level mixed-effects model with sequence as a random intercept.

    This is the formal version of the clustering argument: it estimates the
    between-sequence variance explicitly rather than sidestepping it by
    aggregating. Reported as a supporting analysis -- the headline numbers stay
    sequence-level, because that estimator makes fewer assumptions.
    """
    need = {"truth_angle", "elbow_angle_filtered", "sequence_id", "backend"}
    if not need <= set(frames.columns):
        return
    d = frames.dropna(subset=["truth_angle", "elbow_angle_filtered"]).copy()
    if d.empty:
        return
    d["abs_err"] = (d["elbow_angle_filtered"] - d["truth_angle"]).abs()
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("\n(statsmodels not installed — skipping the mixed model.")
        print(" pip install statsmodels  if you want it. The sequence-level")
        print(" bootstrap above is already sufficient for the rubric.)")
        return

    print("\n=== Supporting analysis: mixed-effects model ===")
    print(f"  {len(d)} frames nested in {d['sequence_id'].nunique()} sequences")
    try:
        m = smf.mixedlm("abs_err ~ C(backend)", d,
                        groups=d["sequence_id"]).fit()
        var_seq = float(m.cov_re.iloc[0, 0])
        var_res = float(m.scale)
        icc = var_seq / (var_seq + var_res) if (var_seq + var_res) else np.nan
        print(f"  between-sequence variance {var_seq:.1f}, "
              f"residual {var_res:.1f}")
        print(f"  ICC = {icc:.3f}  <- share of error variance explained by")
        print("        which sequence a frame came from. A non-trivial ICC is")
        print("        the quantitative justification for clustering.")
        rows.append({"analysis": "mixed_model_icc", "backend": "all",
                     "metric": "icc", "n_sequences": d["sequence_id"].nunique(),
                     "point": icc})
    except Exception as exc:
        print(f"  model did not converge ({exc}); the bootstrap stands alone.")


# --------------------------------------------------------------------------- #
# occlusion
# --------------------------------------------------------------------------- #

def cmd_occlusion(args) -> int:
    """
    Decompose error against annotated body visibility.

    The slide-2 claim is "it is occlusion, not obliquity". That is currently an
    inference from the ordering of four viewpoint means. This measures it
    directly.

    Method. For every frame the estimator was scored on, count how many of Penn
    Action's 13 annotated joints are marked visible. That count is an
    independent, human-supplied measure of how much of the body the camera can
    see -- it does not come from either estimator, so it cannot be circular.
    Restrict to frames where the arm triple is visible (otherwise there is no
    ground-truth angle to score against), then bin by the count and report error
    per bin.

    If error rises as total visibility falls, occlusion is established as the
    driver by a route that does not depend on the viewpoint label at all.
    """
    frames = _require(OUT / "per_frame.csv",
                      "python src/penn_action.py --limit 40",
                      dtype={"sequence_id": str})
    need = {"sequence_id", "backend", "frame", "truth_angle",
            "elbow_angle_filtered"}
    missing = need - set(frames.columns)
    if missing:
        raise SystemExit(f"per_frame.csv is missing columns: {sorted(missing)}")

    frames = frames.copy()
    frames["seq_key"] = frames["sequence_id"].map(seq_key)
    wanted = set(frames["seq_key"])
    print(f"reading visibility for {len(wanted)} sequences from the .mat labels")

    vis_rows = []
    seen = 0
    for lab in P.sequences(None):
        seen += 1
        key = seq_key(lab["sequence_id"])
        if key not in wanted:
            continue
        v = lab["visibility"]
        n_visible = v.sum(axis=1)
        for f in range(v.shape[0]):
            vis_rows.append({"seq_key": key, "frame": f,
                             "n_joints_visible": int(n_visible[f]),
                             "viewpoint": lab["pose"], "action": lab["action"]})
    if not vis_rows:
        raise SystemExit(
            f"read {seen} .mat labels but none matched the {len(wanted)} "
            "sequence ids in per_frame.csv.\n"
            "Check that Data/PennAction/labels/ is the same copy the "
            "validation run used."
        )

    vis = pd.DataFrame(vis_rows)
    d = frames.merge(vis, on=["seq_key", "frame"], how="inner",
                     suffixes=("", "_lab"))
    d = d.dropna(subset=["truth_angle", "elbow_angle_filtered"]).copy()
    if d.empty:
        raise SystemExit("no frames with both a truth angle and an estimate")
    d["abs_err"] = (d["elbow_angle_filtered"] - d["truth_angle"]).abs()

    # Sequence-level aggregation FIRST, then bin. Binning raw frames would
    # reintroduce the independence problem this module is built to avoid.
    per_seq = (d.groupby(["backend", "seq_key", "n_joints_visible"],
                         as_index=False)
                .agg(abs_err=("abs_err", "mean"), n_frames=("abs_err", "size")))

    print("\n=== Error against annotated body visibility ===")
    print("Bins are the number of the 13 annotated joints marked visible.")
    print("Each row aggregates per sequence first, then bootstraps over")
    print("sequence means.\n")
    print(f"{'backend':<12}{'joints vis':>11}{'n seq':>7}{'n frames':>10}"
          f"   mean abs error")
    rows = []
    for (b, k), sub in per_seq.groupby(["backend", "n_joints_visible"]):
        if len(sub) < 2:
            continue
        pt, lo, hi = boot_ci(sub["abs_err"].values)
        print(f"{b:<12}{k:>11}{len(sub):>7}{int(sub['n_frames'].sum()):>10}"
              f"   {fmt_ci(pt, lo, hi)}")
        rows.append({"backend": b, "n_joints_visible": int(k),
                     "n_sequences": len(sub),
                     "n_frames": int(sub["n_frames"].sum()),
                     "mae_deg": pt, "ci_lo": lo, "ci_hi": hi})

    res = pd.DataFrame(rows)
    EXP.mkdir(parents=True, exist_ok=True)
    res.to_csv(EXP / "occlusion_bins.csv", index=False)

    # Correlation across sequences: does lower visibility mean higher error?
    print("\n=== Visibility–error association, across sequences ===")
    for b, sub in per_seq.groupby("backend"):
        agg = (sub.groupby("seq_key")
                  .agg(vis=("n_joints_visible", "mean"),
                       err=("abs_err", "mean")))
        if len(agg) < 4:
            continue
        try:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(agg["vis"], agg["err"])
            print(f"  {b:<12} Spearman rho={rho:+.3f}, p={pval:.4f}, "
                  f"n={len(agg)} sequences")
            print("               negative rho = less of the body visible "
                  "means more error")
            rows.append({"backend": b, "n_joints_visible": -1,
                         "n_sequences": len(agg), "mae_deg": np.nan,
                         "spearman_rho": float(rho), "p_value": float(pval)})
        except Exception as exc:                          # pragma: no cover
            print(f"  {b:<12} (Spearman unavailable: {exc})")

    pd.DataFrame(rows).to_csv(EXP / "occlusion_bins.csv", index=False)
    _plot_occlusion(res)
    print(f"\nWrote {(EXP / 'occlusion_bins.csv').relative_to(ROOT)} and "
          f"{(EXP / 'error_by_visibility.png').relative_to(ROOT)}")
    print("\nFor the thesis: this converts 'it is occlusion, not obliquity'")
    print("from an inference about four means into a measured dose–response")
    print("relationship against a human-supplied visibility count.")
    return 0


def _plot_occlusion(res: pd.DataFrame) -> None:
    if res.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, (b, sub) in enumerate(res.groupby("backend")):
        sub = sub.sort_values("n_joints_visible")
        colour = [FOREST, RED][i % 2]
        yerr = np.vstack([
            (sub["mae_deg"] - sub["ci_lo"]).clip(lower=0).fillna(0),
            (sub["ci_hi"] - sub["mae_deg"]).clip(lower=0).fillna(0),
        ])
        ax.errorbar(sub["n_joints_visible"], sub["mae_deg"], yerr=yerr,
                    marker="o", capsize=3, color=colour, label=b)
    ax.axhline(A.ICC_TOLERANCE_DEG, color=GREY, ls=":", lw=1.4,
               label=f"ICC {A.ICC_TOLERANCE_DEG:.0f}° tolerance")
    ax.set_xlabel("Annotated joints visible (of 13)")
    ax.set_ylabel("Mean absolute elbow-angle error (°)")
    ax.set_title("Error against body visibility — bars are 95% CI over sequences")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    EXP.mkdir(parents=True, exist_ok=True)
    fig.savefig(EXP / "error_by_visibility.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #

def cmd_resolution(args) -> int:
    """
    Error against image resolution.

    Penn Action frames cap at 640x480, which the thesis currently cites as a
    reason the ~20 deg figure is a lower bound for HD broadcast footage. That is
    an assertion. Downscaling the frames and measuring the slope turns it into
    an extrapolation with a stated basis.

    A geometric note that matters: the elbow angle is invariant under uniform
    scaling, so the ground-truth angle does not change when the image is
    resized, and the annotated coordinates need no rescaling. The only thing
    that changes is how much pixel evidence the estimator has. That is what
    makes this a clean single-variable experiment.
    """
    import cv2

    scales = args.scales
    seqs = P.sequences(args.classes)
    if not seqs:
        raise SystemExit(f"no sequences found for classes {args.classes}")
    by_class: dict[str, list[dict]] = {}
    for s in seqs:
        by_class.setdefault(s["action"], []).append(s)
    if args.limit:
        seqs = [s for c in by_class for s in by_class[c][:args.limit]]
    print(f"{len(seqs)} sequences x {len(scales)} scales x "
          f"{len(args.backend)} backends")

    records = []
    tmp_root = Path(tempfile.mkdtemp(prefix="resolution_"))
    try:
        for n, lab in enumerate(seqs, 1):
            # The dataset's limb labels are viewer-relative while the
            # estimators' are anatomical, so the truth and the estimate must be
            # taken from DIFFERENT arm names. This is defect D1. Reintroducing
            # it here silently doubles the error, so the flip is applied through
            # penn_action's own helper rather than reimplemented.
            penn_arm = P.dominant_arm(lab)
            est_arm = (P.estimator_arm_for(penn_arm)
                       if hasattr(P, "estimator_arm_for") else penn_arm)
            truth = P.truth_angles(lab, penn_arm)
            for scale in scales:
                if scale == 1.0:
                    paths = lab["frames"]
                else:
                    d = tmp_root / f"{lab['sequence_id']}_{scale}"
                    d.mkdir(parents=True, exist_ok=True)
                    paths = []
                    for src in lab["frames"]:
                        img = cv2.imread(str(src))
                        if img is None:
                            continue
                        h, w = img.shape[:2]
                        small = cv2.resize(
                            img, (max(int(w * scale), 8), max(int(h * scale), 8)),
                            interpolation=cv2.INTER_AREA)
                        dst = d / src.name
                        cv2.imwrite(str(dst), small)
                        paths.append(dst)
                for backend_name in args.backend:
                    print(f"[{n}/{len(seqs)}] {lab['sequence_id']:<24} "
                          f"scale={scale:<5} {backend_name}", flush=True)
                    try:
                        backend = get_backend(backend_name)
                        try:
                            rows = [fk.as_row(est_arm)
                                    for fk in backend.run_images(paths, est_arm)]
                        finally:
                            backend.close()
                        est = A.compute_angles(pd.DataFrame(rows))
                        j = truth.merge(
                            est[["frame", "elbow_angle_filtered"]],
                            on="frame", how="inner").dropna(
                                subset=["truth_angle", "elbow_angle_filtered"])
                        if j.empty:
                            continue
                        diff = j["elbow_angle_filtered"] - j["truth_angle"]
                        records.append({
                            "sequence_id": lab["sequence_id"],
                            "action": lab["action"],
                            "viewpoint": lab["pose"],
                            "penn_arm": penn_arm,
                            "est_arm": est_arm,
                            "backend": backend_name,
                            "scale": scale,
                            "n_compared": len(j),
                            "mae_deg": float(diff.abs().mean()),
                            "rmse_deg": float(np.sqrt((diff ** 2).mean())),
                            "frac_within_icc": float(
                                (diff.abs() <= A.ICC_TOLERANCE_DEG).mean()),
                        })
                    except Exception as exc:
                        print(f"  !! {type(exc).__name__}: {exc}")
                if scale != 1.0:
                    shutil.rmtree(tmp_root / f"{lab['sequence_id']}_{scale}",
                                  ignore_errors=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    if not records:
        raise SystemExit("nothing scored")

    df = pd.DataFrame(records)
    EXP.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXP / "resolution_sequences.csv", index=False)

    # Restrict to sequences that produced a result at EVERY scale. Otherwise
    # the comparison is confounded: a sequence the estimator fails on at 0.25
    # simply drops out of that row, so the surviving subset is easier, and the
    # scale effect is entangled with which sequences survived. Complete cases
    # make it a within-sequence comparison, which is what the question needs.
    scales_run = sorted(df["scale"].unique())
    counts = df.groupby(["backend", "sequence_id"])["scale"].nunique()
    complete = counts[counts == len(scales_run)].index
    n_before = df["sequence_id"].nunique()
    df_c = df.set_index(["backend", "sequence_id"]).loc[
        df.set_index(["backend", "sequence_id"]).index.isin(complete)
    ].reset_index()
    print(f"\n{df_c['sequence_id'].nunique()} of {n_before} sequences produced "
          f"a result at all {len(scales_run)} scales; the rest are excluded so "
          "the comparison is within-sequence.")
    if df_c.empty:
        df_c = df
        print("  (no complete cases — falling back to all rows, and the "
              "comparison below is NOT paired)")

    print("\n=== Error against image scale ===")
    print("95% CI bootstrapped over sequences at each scale, complete cases only.\n")
    print(f"{'backend':<12}{'scale':>7}{'n seq':>7}   mean abs error")
    summary = []
    for (b, s), sub in df_c.groupby(["backend", "scale"]):
        pt, lo, hi = boot_ci(sub["mae_deg"].values)
        print(f"{b:<12}{s:>7}{len(sub):>7}   {fmt_ci(pt, lo, hi)}")
        summary.append({"backend": b, "scale": s, "n_sequences": len(sub),
                        "mae_deg": pt, "ci_lo": lo, "ci_hi": hi})
    sm = pd.DataFrame(summary)

    # Paired within-sequence deltas against native resolution. This is the
    # actual test: does downscaling THIS sequence make THIS sequence worse?
    top = max(scales_run)
    for b, sub in df_c.groupby("backend"):
        wide = sub.pivot_table(index="sequence_id", columns="scale",
                               values="mae_deg")
        print(f"\n=== Paired change from scale {top}, {b} ===")
        for s in scales_run:
            if s == top or s not in wide.columns:
                continue
            d = (wide[s] - wide[top]).dropna()
            if len(d) < 3:
                continue
            pt, lo, hi = boot_ci(d.values)
            verdict = ("no detectable change" if np.isfinite(lo) and lo <= 0 <= hi
                       else ("worse when downscaled" if pt > 0 else
                             "better when downscaled — investigate"))
            print(f"  scale {s}: {fmt_ci(pt, lo, hi)}  n={len(d)}  → {verdict}")
    sm.to_csv(EXP / "resolution_summary.csv", index=False)

    # Slope, and what it licenses. Deliberately conservative wording: this
    # supports a DIRECTIONAL claim about HD footage, not a numeric prediction.
    print("\n=== Interpretation ===")
    for b, sub in sm.groupby("backend"):
        sub = sub.sort_values("scale")
        if len(sub) < 2:
            continue
        slope = np.polyfit(sub["scale"], sub["mae_deg"], 1)[0]
        print(f"  {b:<12} slope {slope:+.1f}° per unit scale")
        if slope < 0:
            print("               error falls as resolution rises, so the "
                  "640x480 figure is a genuine upper bound on HD error.")
            print("               Report the direction and the slope; do not "
                  "extrapolate a specific HD number from two or three points.")
        else:
            print("               error does NOT fall with resolution. That is "
                  "a stronger finding than the caveat it replaces: the")
            print("               bottleneck is not pixels, so better footage "
                  "will not rescue the method.")
    _plot_resolution(sm)
    print(f"\nWrote {(EXP / 'resolution_summary.csv').relative_to(ROOT)}, "
          f"resolution_sequences.csv and error_by_resolution.png")
    return 0


def _plot_resolution(sm: pd.DataFrame) -> None:
    if sm.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for i, (b, sub) in enumerate(sm.groupby("backend")):
        sub = sub.sort_values("scale")
        colour = [FOREST, RED][i % 2]
        yerr = np.vstack([
            (sub["mae_deg"] - sub["ci_lo"]).clip(lower=0).fillna(0),
            (sub["ci_hi"] - sub["mae_deg"]).clip(lower=0).fillna(0),
        ])
        ax.errorbar(sub["scale"], sub["mae_deg"], yerr=yerr, marker="o",
                    capsize=3, color=colour, label=b)
    ax.axhline(A.ICC_TOLERANCE_DEG, color=GREY, ls=":", lw=1.4,
               label=f"ICC {A.ICC_TOLERANCE_DEG:.0f}° tolerance")
    ax.set_xlabel("Image scale (1.0 = native Penn Action resolution)")
    ax.set_ylabel("Mean absolute elbow-angle error (°)")
    ax.set_title("Error against image resolution — bars are 95% CI over sequences")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    EXP.mkdir(parents=True, exist_ok=True)
    fig.savefig(EXP / "error_by_resolution.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Statistical validity and error-decomposition experiments")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("stats", help="sequence-clustered CIs and tests")
    s1.add_argument("--no-mixed", action="store_true",
                    help="skip the mixed-effects supporting analysis")
    s1.set_defaults(func=cmd_stats)

    s2 = sub.add_parser("occlusion", help="error against annotated visibility")
    s2.set_defaults(func=cmd_occlusion)

    s3 = sub.add_parser("resolution", help="error against image resolution")
    s3.add_argument("--scales", nargs="+", type=float,
                    default=[1.0, 0.5, 0.25])
    s3.add_argument("--backend", nargs="+", default=["mediapipe"])
    s3.add_argument("--classes", nargs="*", default=P.OVERHEAD_CLASSES)
    s3.add_argument("--limit", type=int, default=15,
                    help="sequences per class; keep small, this re-runs inference")
    s3.set_defaults(func=cmd_resolution)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

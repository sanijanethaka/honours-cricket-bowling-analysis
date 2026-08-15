"""
Ground-truth validation on the Penn Action dataset.

    python src/penn_action.py --check                    # verify layout + joint mapping
    python src/penn_action.py --limit 12                 # small pilot run
    python src/penn_action.py --classes baseball_pitch tennis_serve --limit 30

Why this exists
---------------
The cricket experiment can only measure MediaPipe against YOLO11-pose. When they
disagree, nothing in the data says which one is closer to the truth. Penn Action
ships hand-annotated 2D joint positions for every frame, so here the same
pipeline can be scored against a reference rather than against another estimate.

Two questions it answers that the cricket data cannot:

1. What is each estimator's ABSOLUTE elbow-angle error, in degrees?
2. Does that error grow as the camera moves off side-on? Penn Action labels each
   sequence with a coarse viewpoint, which makes this directly testable. The
   planar-geometry assumption in Chapter 3 predicts that it should -- and
   Ruescas-Nicolau et al. (2024) is cited for exactly this claim. Measuring it
   here turns a cited assumption into a reproduced result.

Expected layout after extracting the dataset archive:

    Data/PennAction/
        frames/<sequence_id>/000001.jpg ...
        labels/<sequence_id>.mat

Annotation schema (per the dataset README):
    action     class name, e.g. 'tennis_serve'
    pose       coarse viewpoint, e.g. 'back'
    x, y       [nframes x 13] joint pixel coordinates
    visibility [nframes x 13] logical
    bbox       [nframes x 4]
    train      1 = train, 0 = test
    nframes, dimensions

Coordinates are already in pixels, which is what src/angles.py requires.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import angles as A                    # noqa: E402
from backends import get_backend      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PENN = ROOT / "Data" / "PennAction"
OUT = ROOT / "Outputs" / "penn_action"

#: Penn Action's 13 joints, in the order the x/y columns are stored.
#: NOT verified against a downloaded copy at the time of writing -- run
#: `--check` before trusting any numbers. The checker validates the mapping
#: against body geometry rather than assuming it is right.
JOINT_ORDER = [
    "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

#: Classes whose kinematics resemble a bowling delivery: an overhead or
#: near-overhead arm extension driven from the shoulder. baseball_pitch is the
#: closest analogue to a cricket delivery in the whole dataset.
OVERHEAD_CLASSES = ["baseball_pitch", "tennis_serve"]

#: Contrast group: arm-driven but NOT overhead extension. Note that Penn
#: Action's 'bowling' class is TEN-PIN bowling -- an underarm delivery with no
#: kinematic relationship to cricket bowling. It sits here deliberately, both as
#: a contrast and because the naming collision (cf. BowlingDL, Janbi and
#: Almuaythir 2023) is worth a footnote in Chapter 3. If error is similar here,
#: the finding is about pose estimation generally rather than about overhead
#: extension specifically.
CONTRAST_CLASSES = ["golf_swing", "baseball_swing", "bowling"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def joint_index(name: str) -> int:
    return JOINT_ORDER.index(name)


def load_label(mat_path: Path) -> dict:
    """Read one Penn Action .mat annotation into plain Python types."""
    from scipy.io import loadmat

    m = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    def get(key, default=None):
        return m[key] if key in m else default

    return {
        "sequence_id": mat_path.stem,
        "action": str(get("action", "")),
        "pose": str(get("pose", "")),
        "x": np.atleast_2d(np.asarray(get("x"), dtype=float)),
        "y": np.atleast_2d(np.asarray(get("y"), dtype=float)),
        "visibility": np.atleast_2d(np.asarray(get("visibility"))).astype(bool),
        "train": int(get("train", -1)),
        "nframes": int(get("nframes", 0) or 0),
    }


def sequences(classes: list[str] | None = None) -> list[dict]:
    labels_dir = PENN / "labels"
    if not labels_dir.is_dir():
        raise SystemExit(
            f"Penn Action labels not found at {labels_dir}\n"
            "Extract the dataset so that Data/PennAction/{frames,labels}/ exist."
        )
    out = []
    for mat in sorted(labels_dir.glob("*.mat")):
        try:
            lab = load_label(mat)
        except Exception as exc:
            print(f"  !! could not read {mat.name}: {exc}")
            continue
        if classes and lab["action"] not in classes:
            continue
        frames_dir = PENN / "frames" / lab["sequence_id"]
        if not frames_dir.is_dir():
            continue
        lab["frames"] = sorted(frames_dir.glob("*.jpg"))
        if lab["frames"]:
            out.append(lab)
    return out


# --------------------------------------------------------------------------- #
# Joint-mapping self-check
# --------------------------------------------------------------------------- #

def check_mapping(sample: list[dict]) -> bool:
    """
    Validate JOINT_ORDER against body geometry instead of trusting it.

    Two families of test:

    1. Vertical ordering. In image coordinates y increases downward, so across a
       large sample of upright athletes the mean y should satisfy head <
       shoulder < hip < knee < ankle. These are robust for any action.

    2. Limb rigidity. The upper arm and forearm are rigid segments, so their
       pixel length should be roughly stable across the frames of a sequence
       (modulo foreshortening). If the elbow and wrist columns were swapped, or
       an index were off by one, the "segments" would join unrelated body parts
       and their lengths would fluctuate wildly. This is the real diagnostic.

    Note on the arm: 'elbow above wrist' is NOT a valid check for this dataset.
    Every class here is an overhead or swing action -- tennis serve, baseball
    pitch -- where the wrist sits above the elbow for much of the sequence.
    Averaged over frames the two are level, so the test fails on correct data.
    It is reported below as information only.
    """
    if not sample:
        print("no sequences to check")
        return False

    means = {}
    for name in JOINT_ORDER:
        i = joint_index(name)
        vals = []
        for s_ in sample:
            vis = s_["visibility"][:, i]
            if vis.any():
                vals.append(s_["y"][vis, i].mean())
        means[name] = float(np.mean(vals)) if vals else np.nan

    print("\nmean y (pixels, larger = lower in frame):")
    for name in JOINT_ORDER:
        print(f"  {name:<16} {means[name]:8.1f}")

    # --- limb rigidity -----------------------------------------------------
    def segment_stability(a: str, b: str) -> tuple[float, float]:
        """Median relative spread of |a-b| within a sequence, and median length."""
        spreads, lengths = [], []
        ia, ib = joint_index(a), joint_index(b)
        for s_ in sample:
            vis = s_["visibility"][:, ia] & s_["visibility"][:, ib]
            if vis.sum() < 5:
                continue
            d = np.hypot(s_["x"][vis, ia] - s_["x"][vis, ib],
                         s_["y"][vis, ia] - s_["y"][vis, ib])
            med = float(np.median(d))
            if med <= 1:
                continue
            iqr = float(np.percentile(d, 75) - np.percentile(d, 25))
            spreads.append(iqr / med)
            lengths.append(med)
        if not spreads:
            return np.nan, np.nan
        return float(np.median(spreads)), float(np.median(lengths))

    print("\nlimb rigidity (relative IQR of segment length; lower = more rigid):")
    seg = {}
    for a, b in (("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
                 ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                 ("right_hip", "right_knee"), ("right_knee", "right_ankle")):
        spread, length = segment_stability(a, b)
        seg[(a, b)] = (spread, length)
        print(f"  {a:>15} -> {b:<15} relIQR {spread:5.2f}   median length {length:6.1f} px")

    upper = seg[("right_shoulder", "right_elbow")][1]
    fore = seg[("right_elbow", "right_wrist")][1]
    ratio = fore / upper if upper else np.nan

    checks = [
        ("head above shoulders", means["head"] < means["left_shoulder"]),
        ("shoulder above elbow", means["left_shoulder"] < means["left_elbow"]
                                 and means["right_shoulder"] < means["right_elbow"]),
        ("shoulder above hip", means["left_shoulder"] < means["left_hip"]),
        ("hip above knee", means["left_hip"] < means["left_knee"]),
        ("knee above ankle", means["left_knee"] < means["left_ankle"]),
        ("upper arm is a rigid segment",
         seg[("right_shoulder", "right_elbow")][0] < 0.45),
        ("forearm is a rigid segment",
         seg[("right_elbow", "right_wrist")][0] < 0.45),
        ("forearm length comparable to upper arm", 0.5 < ratio < 1.8),
    ]
    print()
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed

    # Informational only -- fails on correct data for overhead actions.
    arm_note = (means["left_elbow"] < means["left_wrist"]
                and means["right_elbow"] < means["right_wrist"])
    print(f"  [info] wrist below elbow on average: {arm_note} "
          "(expected False for overhead/swing classes)")

    if not ok:
        print(
            "\nJOINT_ORDER in src/penn_action.py does not match this copy of the\n"
            "dataset. Fix the constant before running the validation -- every\n"
            "number downstream depends on it."
        )
    return ok


# --------------------------------------------------------------------------- #
# Angles
# --------------------------------------------------------------------------- #

def truth_angles(lab: dict, arm: str) -> pd.DataFrame:
    """Elbow angle per frame from the human annotation, in pixel space."""
    si, ei, wi = (joint_index(f"{arm}_shoulder"),
                  joint_index(f"{arm}_elbow"),
                  joint_index(f"{arm}_wrist"))
    rows = []
    n = min(len(lab["frames"]), lab["x"].shape[0])
    for f in range(n):
        vis = lab["visibility"][f]
        if not (vis[si] and vis[ei] and vis[wi]):
            rows.append({"frame": f, "truth_angle": np.nan, "truth_visible": False})
            continue
        ang = A.elbow_angle(
            np.array([lab["x"][f, si], lab["y"][f, si]]),
            np.array([lab["x"][f, ei], lab["y"][f, ei]]),
            np.array([lab["x"][f, wi], lab["y"][f, wi]]),
        )
        rows.append({"frame": f, "truth_angle": ang, "truth_visible": True})
    return pd.DataFrame(rows)


def dominant_arm(lab: dict) -> str:
    """
    Pick the arm to analyse: the one annotated visible in more frames.

    Penn Action does not label handedness. Choosing by visibility is a stated
    heuristic, not a fact about the athlete, and it is recorded in the output so
    the choice is auditable.
    """
    counts = {}
    for arm in ("left", "right"):
        idxs = [joint_index(f"{arm}_{p}") for p in ("shoulder", "elbow", "wrist")]
        counts[arm] = int(lab["visibility"][:, idxs].all(axis=1).sum())
    return max(counts, key=counts.get)


def estimator_angles(lab: dict, backend_name: str, arm: str) -> pd.DataFrame:
    backend = get_backend(backend_name)
    try:
        rows = [fk.as_row(arm) for fk in backend.run_images(lab["frames"], arm)]
    finally:
        backend.close()
    return A.compute_angles(pd.DataFrame(rows))


def score(lab: dict, backend_name: str) -> tuple[dict, pd.DataFrame]:
    arm = dominant_arm(lab)
    truth = truth_angles(lab, arm)
    est = estimator_angles(lab, backend_name, arm)

    j = truth.merge(est[["frame", "elbow_angle_filtered", "elbow_angle_raw"]],
                    on="frame", how="inner")
    # Compare only where BOTH the annotation and the estimator have a usable
    # measurement. Frames the estimator's confidence filter rejected are not
    # errors -- they are refusals, and are reported separately as coverage.
    both = j.dropna(subset=["truth_angle", "elbow_angle_filtered"])
    diff = both["elbow_angle_filtered"] - both["truth_angle"]

    rec = {
        "sequence_id": lab["sequence_id"],
        "action": lab["action"],
        "viewpoint": lab["pose"],
        "arm_analysed": arm,
        "train": lab["train"],
        "backend": backend_name,
        "n_frames": len(j),
        "n_truth_visible": int(j["truth_visible"].sum()),
        "n_compared": len(both),
        "coverage": float(len(both) / max(int(j["truth_visible"].sum()), 1)),
    }
    if both.empty:
        rec.update({k: np.nan for k in
                    ("mae_deg", "median_ae_deg", "p95_ae_deg", "max_ae_deg",
                     "bias_deg", "rmse_deg", "frac_within_icc", "pearson_r")})
    else:
        rec.update({
            "mae_deg": float(diff.abs().mean()),
            "median_ae_deg": float(diff.abs().median()),
            "p95_ae_deg": float(diff.abs().quantile(0.95)),
            "max_ae_deg": float(diff.abs().max()),
            "bias_deg": float(diff.mean()),
            "rmse_deg": float(np.sqrt((diff ** 2).mean())),
            "frac_within_icc": float((diff.abs() <= A.ICC_TOLERANCE_DEG).mean()),
            "pearson_r": (float(both["elbow_angle_filtered"].corr(both["truth_angle"]))
                          if len(both) > 2 else np.nan),
        })
    return rec, j


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def plot_error_by_viewpoint(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.dropna(subset=["mae_deg"])
    if d.empty:
        return
    FOREST, RED, GREY = "#2C5F2D", "#B03A2E", "#5A6B5B"
    views = sorted(d["viewpoint"].unique())
    backends = sorted(d["backend"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    palette = [FOREST, RED]
    w = 0.8 / max(len(backends), 1)
    for i, b in enumerate(backends):
        sub = d[d["backend"] == b]
        means = [sub[sub["viewpoint"] == v]["mae_deg"].mean() for v in views]
        errs = [sub[sub["viewpoint"] == v]["mae_deg"].std(ddof=0) for v in views]
        ax1.bar(np.arange(len(views)) + i * w, means, w, yerr=errs, capsize=3,
                color=palette[i % 2], label=b)
    ax1.axhline(A.ICC_TOLERANCE_DEG, color=GREY, ls=":", lw=1.4,
                label=f"ICC {A.ICC_TOLERANCE_DEG:.0f}° tolerance")
    ax1.set_xticks(np.arange(len(views)) + w * (len(backends) - 1) / 2)
    ax1.set_xticklabels(views)
    ax1.set_xlabel("Annotated camera viewpoint")
    ax1.set_ylabel("Mean absolute elbow-angle error (°)")
    ax1.set_title("Error vs ground truth, by viewpoint")
    ax1.grid(alpha=0.3, axis="y")
    ax1.legend(fontsize=9)

    acts = sorted(d["action"].unique())
    for i, b in enumerate(backends):
        sub = d[d["backend"] == b]
        means = [sub[sub["action"] == a]["mae_deg"].mean() for a in acts]
        ax2.bar(np.arange(len(acts)) + i * w, means, w, color=palette[i % 2], label=b)
    ax2.axhline(A.ICC_TOLERANCE_DEG, color=GREY, ls=":", lw=1.4)
    ax2.set_xticks(np.arange(len(acts)) + w * (len(backends) - 1) / 2)
    ax2.set_xticklabels(acts, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Mean absolute elbow-angle error (°)")
    ax2.set_title("Error vs ground truth, by action class")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "error_by_viewpoint.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Penn Action ground-truth validation")
    ap.add_argument("--check", action="store_true",
                    help="verify dataset layout and joint mapping, then exit")
    ap.add_argument("--classes", nargs="*", default=None,
                    help=f"default: {OVERHEAD_CLASSES + CONTRAST_CLASSES}")
    ap.add_argument("--backend", nargs="+", default=["mediapipe", "yolo11pose"])
    ap.add_argument("--limit", type=int, default=None,
                    help="max sequences per class (keeps runtime sane)")
    args = ap.parse_args(argv)

    classes = args.classes if args.classes is not None else (
        OVERHEAD_CLASSES + CONTRAST_CLASSES)
    seqs = sequences(classes)
    if not seqs:
        raise SystemExit(f"no sequences found for classes {classes}")

    by_class: dict[str, list[dict]] = {}
    for s in seqs:
        by_class.setdefault(s["action"], []).append(s)
    if args.limit:
        seqs = [s for c in by_class for s in by_class[c][:args.limit]]

    print(f"{len(seqs)} sequences across {len(by_class)} classes")
    for c, items in sorted(by_class.items()):
        n = min(len(items), args.limit) if args.limit else len(items)
        views = sorted({i["pose"] for i in items})
        print(f"  {c:<16} {n:>4} sequences   viewpoints: {views}")

    if args.check:
        ok = check_mapping(seqs[:200])
        print("\nlayout OK, mapping " + ("verified" if ok else "SUSPECT"))
        return 0 if ok else 1

    if not check_mapping(seqs[:200]):
        raise SystemExit("aborting: fix JOINT_ORDER first")

    records, frame_tables = [], []
    for n, lab in enumerate(seqs, 1):
        for backend in args.backend:
            print(f"[{n}/{len(seqs)}] {lab['sequence_id']} {lab['action']:<16} "
                  f"view={lab['pose']:<8} {backend}", flush=True)
            try:
                rec, per_frame = score(lab, backend)
            except Exception as exc:
                print(f"  !! {type(exc).__name__}: {exc}")
                continue
            records.append(rec)
            per_frame["sequence_id"] = lab["sequence_id"]
            per_frame["backend"] = backend
            frame_tables.append(per_frame)

    if not records:
        raise SystemExit("no sequences scored")

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUT / "sequence_scores.csv", index=False)
    pd.concat(frame_tables, ignore_index=True).to_csv(
        OUT / "per_frame.csv", index=False)

    summary = (df.groupby(["backend", "viewpoint"])
                 .agg(n_sequences=("sequence_id", "count"),
                      mae_deg=("mae_deg", "mean"),
                      rmse_deg=("rmse_deg", "mean"),
                      bias_deg=("bias_deg", "mean"),
                      frac_within_icc=("frac_within_icc", "mean"),
                      coverage=("coverage", "mean"))
                 .reset_index())
    summary.to_csv(OUT / "summary_by_viewpoint.csv", index=False)
    plot_error_by_viewpoint(df)

    print("\n=== mean absolute elbow-angle error vs human annotation ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nWrote {OUT.relative_to(ROOT)}/"
          "{sequence_scores.csv, per_frame.csv, summary_by_viewpoint.csv, "
          "error_by_viewpoint.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

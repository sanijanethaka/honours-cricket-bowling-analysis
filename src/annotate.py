"""
Reference-annotation test-retest, for the annotation noise floor.

    python src/annotate.py --round 1        # do this today
    python src/annotate.py --round 2        # do this in a week, no peeking
    python src/annotate.py --compare        # the number for Table 4.8

Why this matters
----------------
Every error figure in Chapter 4 is measured against Penn Action's human
annotation. No estimator can be more accurate than the reference it is scored
against, so the reference's own repeatability is a ceiling on the whole study.
If your test-retest error is 8 deg, then a 19.7 deg estimator error means
something quite different from what it means if your test-retest is 2 deg.

Viewpoint (4.5.4) and occlusion (4.6.1) both came back null, so this is the
leading remaining candidate explanation for the residual error. It is also the
only experiment here with a hard scheduling dependency: the two rounds must be
separated by roughly a week so that round two is not simply recall of round one.

How the design avoids fooling you
---------------------------------
* Both rounds annotate the SAME frames, chosen by a fixed seed, because a
  test-retest comparison is meaningless across different images.
* Round two never displays round one. The file is not read until --compare.
* Frames are presented in a different, seeded order each round, so position in
  the sequence cannot cue recall.
* Only frames where all three arm joints are annotated visible are eligible,
  which is the same condition under which the estimators were scored.

Clicking
--------
Three clicks per frame, in this order: SHOULDER, ELBOW, WRIST -- of the arm
named in the window title. Middle-click or backspace undoes the last click;
close the window to stop early (progress is saved).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import angles as A                    # noqa: E402
import penn_action as P               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Outputs" / "annotation"
SEED = 4486373                        # fixed: both rounds must see the same frames


# --------------------------------------------------------------------------- #

def eligible_frames(n: int, classes: list[str] | None) -> pd.DataFrame:
    """
    Sample n frames in which all three arm joints of the dominant arm are
    annotated visible -- the same condition under which the estimators were
    scored, so the noise floor is measured on comparable material.
    """
    seqs = P.sequences(classes)
    if not seqs:
        raise SystemExit(f"no sequences found for classes {classes}")

    pool = []
    for lab in seqs:
        arm = P.dominant_arm(lab)
        idxs = [P.joint_index(f"{arm}_{p}") for p in ("shoulder", "elbow", "wrist")]
        vis = lab["visibility"][:, idxs].all(axis=1)
        n_frames = min(len(lab["frames"]), lab["x"].shape[0], len(vis))
        for f in range(n_frames):
            if vis[f]:
                pool.append({
                    "sequence_id": lab["sequence_id"],
                    "action": lab["action"],
                    "viewpoint": lab["pose"],
                    "arm": arm,
                    "frame": f,
                    "path": str(lab["frames"][f]),
                    "truth_angle": A.elbow_angle(
                        np.array([lab["x"][f, idxs[0]], lab["y"][f, idxs[0]]]),
                        np.array([lab["x"][f, idxs[1]], lab["y"][f, idxs[1]]]),
                        np.array([lab["x"][f, idxs[2]], lab["y"][f, idxs[2]]]),
                    ),
                })
    if not pool:
        raise SystemExit("no eligible frames found")

    df = pd.DataFrame(pool)
    # One frame per sequence at most: 25 frames from 25 different athletes is a
    # broader test of repeatability than 25 frames from three clips.
    df = df.sample(frac=1.0, random_state=SEED).drop_duplicates("sequence_id")
    if len(df) < n:
        print(f"only {len(df)} eligible sequences; using all of them")
        n = len(df)
    return df.head(n).reset_index(drop=True)


def annotate(round_no: int, n: int, classes) -> int:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"round{round_no}.csv"
    if target.exists():
        done = pd.read_csv(target, dtype={"sequence_id": str})
        print(f"{target.name} already has {len(done)} rows; resuming.")
    else:
        done = pd.DataFrame()

    frames = eligible_frames(n, classes)
    already = set(zip(done.get("sequence_id", []), done.get("frame", []))) \
        if len(done) else set()

    # Different presentation order per round so position cannot cue recall.
    order = frames.sample(frac=1.0, random_state=SEED + round_no).reset_index(drop=True)

    print(f"\nRound {round_no}: {len(order)} frames, "
          f"{len(already)} already done.")
    print("Click SHOULDER, then ELBOW, then WRIST. Backspace undoes. "
          "Close the window to stop.\n")

    rows = done.to_dict("records") if len(done) else []
    plt.rcParams["toolbar"] = "none"

    for i, r in order.iterrows():
        if (r["sequence_id"], r["frame"]) in already:
            continue
        img = mpimg.imread(r["path"])
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(img)
        ax.set_axis_off()
        # Penn Action's limb names are viewer-relative, not anatomical (this is
        # the mirrored convention established in penn_action.py). Saying "left
        # arm" is therefore ambiguous to a human annotator, and an annotator who
        # resolves it differently between rounds inflates the test-retest error
        # with a labelling disagreement rather than a pointing one. State the
        # side of the IMAGE instead, which is unambiguous.
        is_left = r["arm"] == "left"
        side = "LEFT" if is_left else "RIGHT"
        ax.set_title(
            f"[{i + 1}/{len(order)}]  annotate the arm on the {side} of the "
            f"IMAGE (your left/right, not the athlete's)\n"
            f"click: shoulder -> elbow -> wrist",
            fontsize=12)

        # Shade the half of the frame the target arm belongs to. The title alone
        # proved ambiguous in practice, and a mis-identified limb does not
        # register as a small error -- it produced disagreements above 100 deg.
        # The shading is an orientation cue, not a bounding box: an arm swung
        # across the body can legitimately end up in the other half.
        w_img = img.shape[1]
        ax.axvspan(0 if is_left else w_img / 2,
                   w_img / 2 if is_left else w_img,
                   color="#FFD54F", alpha=0.13, zorder=2)
        ax.text(0.02 if is_left else 0.98, 0.02, f"THIS SIDE  ({side})",
                transform=ax.transAxes, fontsize=13, fontweight="bold",
                color="#7A5000", ha="left" if is_left else "right",
                bbox=dict(boxstyle="round,pad=0.4", fc="#FFF3CD", ec="#E0A030"))
        fig.tight_layout()
        try:
            pts = plt.ginput(3, timeout=0, show_clicks=True)
        except Exception:
            pts = []
        plt.close(fig)

        if len(pts) < 3:
            print("stopped early — progress saved")
            break

        s, e, w = (np.array(pt, dtype=float) for pt in pts)
        rows.append({
            "sequence_id": r["sequence_id"], "frame": int(r["frame"]),
            "action": r["action"], "viewpoint": r["viewpoint"], "arm": r["arm"],
            "shoulder_x": s[0], "shoulder_y": s[1],
            "elbow_x": e[0], "elbow_y": e[1],
            "wrist_x": w[0], "wrist_y": w[1],
            "my_angle": A.elbow_angle(s, e, w),
            "penn_angle": float(r["truth_angle"]),
        })
        pd.DataFrame(rows).to_csv(target, index=False)
        print(f"  {len(rows):>3} saved   angle {rows[-1]['my_angle']:6.1f}°")

    print(f"\nWrote {target.relative_to(ROOT)} ({len(rows)} frames)")
    if round_no == 1:
        print("\nNow wait about a week before round 2. Put it in your calendar —")
        print("the gap is the whole point, and it is the only part you cannot")
        print("compress later.")
    return 0


def compare() -> int:
    r1p, r2p = OUT / "round1.csv", OUT / "round2.csv"
    for path in (r1p, r2p):
        if not path.is_file():
            raise SystemExit(f"missing {path.relative_to(ROOT)} — "
                             "run both rounds first")
    r1 = pd.read_csv(r1p, dtype={"sequence_id": str})
    r2 = pd.read_csv(r2p, dtype={"sequence_id": str})

    j = r1.merge(r2, on=["sequence_id", "frame"], suffixes=("_1", "_2"))
    if j.empty:
        raise SystemExit("the two rounds share no frames")

    d = (j["my_angle_1"] - j["my_angle_2"]).abs()
    from experiments import boot_ci, fmt_ci
    pt, lo, hi = boot_ci(d.values)

    print(f"\n=== Annotation noise floor, author test-retest ===")
    print(f"  {len(j)} frames annotated twice\n")
    print(f"  mean absolute difference   {fmt_ci(pt, lo, hi)}")
    print(f"  median                     {d.median():.1f}°")
    print(f"  95th percentile            {d.quantile(0.95):.1f}°")
    print(f"  max                        {d.max():.1f}°")

    # Agreement with Penn Action's own annotators is a second, independent
    # quantity: it bounds how far the reference itself may sit from any other
    # competent annotator, which is what actually limits the study.
    v1 = (j["my_angle_1"] - j["penn_angle_1"]).abs()
    print(f"\n  vs Penn Action's annotation, round 1  "
          f"{fmt_ci(*boot_ci(v1.values))}")

    print("\n=== What this licenses ===")
    print(f"  No estimator scored against this reference can do better than")
    print(f"  roughly {pt:.1f}°, because that is the reference's own repeatability.")
    if np.isfinite(pt):
        share = 100 * pt / 19.7
        print(f"  That is about {share:.0f}% of the {19.7:.1f}° MediaPipe error.")
        if share < 30:
            print("  So most of the observed error is NOT annotation noise, and")
            print("  the residual remains unexplained — report that plainly.")
        else:
            print("  So a substantial share of the observed error is attributable")
            print("  to the reference rather than to the estimators.")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "n_frames": len(j), "mae_deg": pt, "ci_lo": lo, "ci_hi": hi,
        "median_deg": float(d.median()), "p95_deg": float(d.quantile(0.95)),
        "max_deg": float(d.max()),
        "vs_penn_mae_deg": float(v1.mean()),
    }]).to_csv(OUT / "noise_floor.csv", index=False)
    j.to_csv(OUT / "paired_annotations.csv", index=False)
    print(f"\nWrote {(OUT / 'noise_floor.csv').relative_to(ROOT)} — "
          "this is Table 4.8.")
    return 0


def review(round_no: int, threshold: float) -> int:
    """
    Display the frames where your annotation and Penn Action's disagree most,
    with both sets of joints drawn on the image.

    A large disagreement has three possible causes and they are not equivalent:
    you mis-identified the limb, you pointed imprecisely, or the reference
    annotation is itself wrong. Only the third is a finding about the dataset,
    and the only way to tell them apart is to look. Do this before quoting any
    agreement figure -- and note that any Penn Action errors you find are
    reportable in their own right, since the whole study is scored against
    that reference.
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    path = OUT / f"round{round_no}.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path.relative_to(ROOT)}")
    d = pd.read_csv(path, dtype={"sequence_id": str})
    d["diff"] = (d["my_angle"] - d["penn_angle"]).abs()
    flagged = d[d["diff"] > threshold].sort_values("diff", ascending=False)
    if flagged.empty:
        print(f"nothing disagrees by more than {threshold:.0f}°")
        return 0

    print(f"\n{len(flagged)} of {len(d)} frames disagree by more than "
          f"{threshold:.0f}°.")
    print("Close each window to move on. Decide for each: my error, or the "
          "dataset's?\n")

    labels = {lab["sequence_id"]: lab for lab in P.sequences(None)}
    for _, r in flagged.iterrows():
        lab = labels.get(r["sequence_id"])
        if lab is None:
            continue
        f = int(r["frame"])
        if f >= len(lab["frames"]):
            continue
        arm = r["arm"]
        idxs = [P.joint_index(f"{arm}_{p}") for p in ("shoulder", "elbow", "wrist")]
        px = [lab["x"][f, i] for i in idxs]
        py = [lab["y"][f, i] for i in idxs]
        mx = [r["shoulder_x"], r["elbow_x"], r["wrist_x"]]
        my = [r["shoulder_y"], r["elbow_y"], r["wrist_y"]]

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(mpimg.imread(lab["frames"][f]))
        ax.plot(px, py, "-o", color="#B03A2E", lw=2.5, ms=9,
                label=f"Penn Action  {r['penn_angle']:.1f}°")
        ax.plot(mx, my, "-o", color="#2C5F2D", lw=2.5, ms=9,
                label=f"mine  {r['my_angle']:.1f}°")
        ax.set_axis_off()
        ax.legend(loc="upper right", fontsize=11)
        ax.set_title(f"{lab['sequence_id']}  {lab['action']}  "
                     f"view={lab['pose']}  arm={arm}  Δ={r['diff']:.1f}°",
                     fontsize=12)
        fig.tight_layout()
        plt.show()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Annotation noise-floor test-retest")
    ap.add_argument("--round", type=int, choices=(1, 2))
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--review", action="store_true",
                    help="inspect frames where you and Penn Action disagree")
    ap.add_argument("--threshold", type=float, default=30.0)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--classes", nargs="*", default=P.OVERHEAD_CLASSES)
    args = ap.parse_args(argv)

    if args.compare:
        return compare()
    if args.review:
        return review(args.round or 1, args.threshold)
    if not args.round:
        ap.error("give --round 1, --round 2, --compare, or --review")
    return annotate(args.round, args.n, args.classes)


if __name__ == "__main__":
    raise SystemExit(main())

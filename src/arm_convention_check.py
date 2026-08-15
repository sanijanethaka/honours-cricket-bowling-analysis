"""
Is Penn Action's left/right ANATOMICAL (the subject's own) or IMAGE-based
(the viewer's)?

This matters because MediaPipe and COCO/YOLO are anatomical: their
"left_shoulder" is the shoulder on the subject's own left. If Penn Action's
labels are viewer-based instead, then for every clip where the subject faces the
camera we would be comparing the truth for one arm against the estimate for the
other -- producing exactly the pattern seen in the first run: large errors, a
bias that flips sign between sequences, and error that tracks viewpoint rather
than difficulty.

    python src/arm_convention_check.py            # Test 1 only (instant)
    python src/arm_convention_check.py --full 12  # adds Test 2 (needs MediaPipe)

Test 1 -- annotations only, no model.
    dx = mean( x[left_shoulder] - x[right_shoulder] )
    If labels are ANATOMICAL, a subject facing the camera has their own left on
    the IMAGE RIGHT, so dx > 0 for 'front' and dx < 0 for 'back'. The sign flips
    with viewpoint.
    If labels are IMAGE-based, dx < 0 for every viewpoint -- no flip.

Test 2 -- the decisive one. Runs MediaPipe on each arm and scores it against
each annotated arm, giving a 2x2 error matrix. If the off-diagonal
(truth-left vs estimate-right, and vice versa) is markedly lower than the
diagonal, the conventions disagree and src/penn_action.py must flip one side.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import angles as A                     # noqa: E402
import penn_action as P                # noqa: E402
from backends import get_backend       # noqa: E402


def test_shoulder_side(seqs) -> None:
    li, ri = P.joint_index("left_shoulder"), P.joint_index("right_shoulder")
    rows = []
    for s in seqs:
        vis = s["visibility"][:, li] & s["visibility"][:, ri]
        if vis.sum() < 5:
            continue
        dx = float(np.mean(s["x"][vis, li] - s["x"][vis, ri]))
        rows.append({"viewpoint": s["pose"], "dx": dx})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no usable sequences")
        return

    g = df.groupby("viewpoint")["dx"].agg(["count", "mean", "median"]).round(2)
    print("\nTest 1 -- mean x(left_shoulder) - x(right_shoulder), in pixels")
    print("  positive = the joint labelled 'left' sits on the IMAGE RIGHT\n")
    print(g.to_string())

    front = df[df.viewpoint == "front"]["dx"].median() if (df.viewpoint == "front").any() else np.nan
    back = df[df.viewpoint == "back"]["dx"].median() if (df.viewpoint == "back").any() else np.nan
    print()
    if np.isnan(front) or np.isnan(back):
        print("  inconclusive: need both 'front' and 'back' sequences")
        return
    if front * back < 0:
        print(f"  front {front:+.1f} vs back {back:+.1f} -- SIGN FLIPS")
        print("  => labels are ANATOMICAL, same convention as MediaPipe/YOLO.")
        print("     No change needed; the large errors have another cause.")
    else:
        print(f"  front {front:+.1f} vs back {back:+.1f} -- NO SIGN FLIP")
        print("  => labels are IMAGE-BASED (viewer's left/right).")
        print("     This DOES NOT match MediaPipe/YOLO, which are anatomical.")
        print("     src/penn_action.py is comparing the wrong arms on roughly")
        print("     half the dataset. Fix before reporting any error figure.")


def test_cross_arm(seqs, limit: int) -> None:
    print(f"\nTest 2 -- 2x2 error matrix over {min(limit, len(seqs))} sequences "
          "(MediaPipe only)\n")
    backend = get_backend("mediapipe")
    cells = {(t, e): [] for t in ("left", "right") for e in ("left", "right")}
    try:
        for n, lab in enumerate(seqs[:limit], 1):
            print(f"  [{n}/{min(limit, len(seqs))}] {lab['sequence_id']} "
                  f"{lab['action']} view={lab['pose']}", flush=True)
            est = {}
            for arm in ("left", "right"):
                rows = [fk.as_row(arm) for fk in backend.run_images(lab["frames"], arm)]
                est[arm] = A.compute_angles(pd.DataFrame(rows))
            for t_arm in ("left", "right"):
                truth = P.truth_angles(lab, t_arm)
                for e_arm in ("left", "right"):
                    j = truth.merge(est[e_arm][["frame", "elbow_angle_filtered"]],
                                    on="frame", how="inner").dropna(
                        subset=["truth_angle", "elbow_angle_filtered"])
                    if len(j) >= 5:
                        cells[(t_arm, e_arm)].append(
                            float((j["elbow_angle_filtered"] - j["truth_angle"]).abs().mean()))
    finally:
        backend.close()

    print("\n  mean absolute error (degrees)")
    print(f"  {'':<18}{'est left':>12}{'est right':>12}")
    for t in ("left", "right"):
        vals = []
        for e in ("left", "right"):
            v = np.mean(cells[(t, e)]) if cells[(t, e)] else np.nan
            vals.append(v)
        print(f"  truth {t:<12}{vals[0]:>12.1f}{vals[1]:>12.1f}")

    diag = np.nanmean([np.mean(cells[("left", "left")] or [np.nan]),
                       np.mean(cells[("right", "right")] or [np.nan])])
    off = np.nanmean([np.mean(cells[("left", "right")] or [np.nan]),
                      np.mean(cells[("right", "left")] or [np.nan])])
    print(f"\n  same-name pairing  {diag:.1f} deg")
    print(f"  swapped pairing    {off:.1f} deg")
    if off < diag - 5:
        print("\n  => SWAPPED pairing is clearly better. The two conventions")
        print("     disagree; flip the arm mapping in src/penn_action.py.")
    elif diag < off - 5:
        print("\n  => Same-name pairing is better. Conventions agree; the large")
        print("     errors are genuine, not a mapping artefact.")
    else:
        print("\n  => Too close to call. Neither pairing is clearly right, which")
        print("     suggests the error is dominated by something else.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", type=int, default=0,
                    help="also run Test 2 on this many sequences")
    ap.add_argument("--classes", nargs="*",
                    default=["baseball_pitch", "baseball_swing"])
    args = ap.parse_args(argv)

    seqs = P.sequences(args.classes)
    print(f"{len(seqs)} sequences loaded from {args.classes}")
    test_shoulder_side(seqs)
    if args.full:
        test_cross_arm(seqs, args.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

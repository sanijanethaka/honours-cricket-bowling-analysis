"""
Batch pipeline runner.

    python src/pipeline.py --all --backend mediapipe yolo11pose
    python src/pipeline.py --clip cummins --backend yolo11pose --debug-video

Every clip in Data/clips.csv is processed through the same six stages with the
same parameters. Nothing is hardcoded per clip: the bowling arm, style and
inclusion status all come from the registry, which is the fix for the class of
error where a left-arm bowler was processed with BOWLING_ARM = "right" and the
resulting signal described the wrong limb.

Excluded clips are still processed (with --include-excluded) so the exclusion
register is backed by regenerable evidence rather than by a stale note.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import angles as A          # noqa: E402
from backends import get_backend  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "Data" / "clips.csv"
CLIP_DIR = ROOT / "Data" / "Clips"
OUT_DIR = ROOT / "Outputs" / "runs"


def load_registry() -> pd.DataFrame:
    if not REGISTRY.exists():
        raise FileNotFoundError(f"clip registry missing: {REGISTRY}")
    df = pd.read_csv(REGISTRY)
    required = {"clip_id", "bowler", "style", "bowling_arm", "file", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"registry missing columns: {sorted(missing)}")
    bad = df.loc[~df["bowling_arm"].isin(["left", "right"]), "clip_id"].tolist()
    if bad:
        raise ValueError(f"bowling_arm must be left/right; offending clips: {bad}")
    return df


def video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "n_frames_reported": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    meta["aspect"] = round(meta["width"] / meta["height"], 4) if meta["height"] else None
    return meta


def plot_signal(df: pd.DataFrame, title: str, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["frame"], df["elbow_angle_raw"], color="0.8", lw=1,
            label="Raw (all detected frames)")
    ax.plot(df["frame"], df["elbow_angle_smoothed"], color="#2C5F2D", lw=2.2,
            label=f"Smoothed (conf ≥ {A.CONF_THRESHOLD})")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Elbow angle (degrees)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run_one(row: pd.Series, backend_name: str, debug_video: bool = False) -> dict:
    clip_path = CLIP_DIR / row["file"]
    meta = video_meta(clip_path)
    out = OUT_DIR / row["clip_id"] / backend_name
    out.mkdir(parents=True, exist_ok=True)

    backend = get_backend(backend_name)
    try:
        rows = [fk.as_row(row["bowling_arm"])
                for fk in backend.run(str(clip_path), row["bowling_arm"])]
    finally:
        backend.close()

    kp = pd.DataFrame(rows)
    kp.to_csv(out / "keypoints.csv", index=False)

    ang = A.compute_angles(kp)
    ang.to_csv(out / "angles.csv", index=False)

    plot_signal(
        ang,
        f"{row['bowler']} — elbow angle, {backend_name} "
        f"({row['bowling_arm']} arm)",
        out / "angle_plot.png",
    )

    feats = A.features(ang, fps=meta["fps"])
    record = {
        "clip_id": row["clip_id"], "bowler": row["bowler"], "style": row["style"],
        "category": row.get("category", ""), "bowling_arm": row["bowling_arm"],
        "status": row["status"], "backend": backend_name,
        **meta, **feats,
    }
    (out / "run.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Elbow-angle pipeline over the clip registry")
    ap.add_argument("--clip", nargs="*", help="clip_id(s); default all included clips")
    ap.add_argument("--all", action="store_true", help="every included clip")
    ap.add_argument("--backend", nargs="+", default=["mediapipe"],
                    help="one or more of: mediapipe yolo11pose")
    ap.add_argument("--include-excluded", action="store_true",
                    help="also process clips marked excluded (regenerates exclusion evidence)")
    ap.add_argument("--debug-video", action="store_true")
    args = ap.parse_args(argv)

    reg = load_registry()
    if args.clip:
        sel = reg[reg["clip_id"].isin(args.clip)]
        unknown = set(args.clip) - set(sel["clip_id"])
        if unknown:
            raise SystemExit(f"unknown clip_id(s): {sorted(unknown)}")
    else:
        sel = reg if args.include_excluded else reg[reg["status"] == "included"]
    if not args.include_excluded and not args.clip:
        sel = sel[sel["status"] == "included"]

    records = []
    for _, row in sel.iterrows():
        for backend in args.backend:
            print(f"[run] {row['clip_id']:<10} {backend:<11} "
                  f"arm={row['bowling_arm']:<5} status={row['status']}", flush=True)
            try:
                records.append(run_one(row, backend, args.debug_video))
            except Exception as exc:  # keep the batch alive; a bad clip is data
                print(f"  !! failed: {type(exc).__name__}: {exc}", flush=True)
                records.append({
                    "clip_id": row["clip_id"], "backend": backend,
                    "status": row["status"], "error": f"{type(exc).__name__}: {exc}",
                })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    features_csv = ROOT / "Outputs" / "features.csv"
    if features_csv.exists():
        prior = pd.read_csv(features_csv)
        key = ["clip_id", "backend"]
        prior = prior[~prior.set_index(key).index.isin(df.set_index(key).index)]
        df = pd.concat([prior, df], ignore_index=True)
    df.sort_values(["clip_id", "backend"]).to_csv(features_csv, index=False)
    print(f"\nWrote {features_csv.relative_to(ROOT)}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

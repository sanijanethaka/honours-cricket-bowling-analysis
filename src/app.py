"""
Elbow-angle analysis — demonstration interface.

    pip install gradio
    python src/app.py

Then open the http://127.0.0.1:7860 link it prints.

What this is
------------
A thin front end over the existing pipeline. It adds no new analysis: it calls
the same backends.py and angles.py that produced every number in Chapter 4, so
what the interface shows and what the thesis reports cannot drift apart. That is
the point of building it this way rather than as a separate script.

What it deliberately does NOT do
--------------------------------
It does not judge whether a delivery is legal. The validation in Chapter 4 put
the measurement error at roughly +/-20 degrees against hand-annotated ground
truth, which is larger than the 15 degree tolerance the Laws of Cricket use to
separate a legal delivery from a throw. An interface that displayed a
legal/illegal verdict would be claiming a precision the measurement does not
have, and would contradict the scope commitment in Section 1.4. The disclaimer
below is therefore part of the artefact, not decoration around it.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import angles as A                    # noqa: E402
from backends import get_backend      # noqa: E402

FOREST, RED, GREY = "#2C5F2D", "#B03A2E", "#5A6B5B"

DISCLAIMER = """
### What this tool does and does not do

**Does:** extracts shoulder, elbow and wrist positions from the video, computes
the elbow angle in each frame, smooths the result and reports simple movement
features.

**Does not:** decide whether a delivery is legal. Validation against
hand-annotated ground truth (160 sequences, 12,196 frames) put this method's
mean absolute error at **19.7&deg;** for MediaPipe and **21.0&deg;** for
YOLO11-pose. Both exceed the **15&deg;** ICC tolerance that separates a legal
delivery from a throw, so the output describes the *shape* of an action and
flags deliveries worth closer inspection. It is not a legality verdict and not a
clinical or selection assessment.
"""


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #

def _video_meta(path: str) -> tuple[float, int, int, int]:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return float(fps), w, h, n


def _overlay_video(src_path: str, rows: pd.DataFrame, out_path: str,
                   fps: float, size: tuple[int, int]) -> str:
    """
    Redraw the clip with the analysed arm marked on each frame.

    Frames the confidence filter rejected are drawn in red and labelled, rather
    than left blank. A viewer should be able to see WHERE the method refused to
    measure, because coverage is reported separately from error throughout the
    thesis and the same distinction should be visible here.
    """
    import cv2

    cap = cv2.VideoCapture(src_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, size)
    by_frame = {int(r["frame"]): r for _, r in rows.iterrows()}

    green, red, white = (45, 95, 44), (46, 58, 176), (255, 255, 255)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            r = by_frame.get(idx)
            if r is not None and not pd.isna(r.get("elbow_angle_raw", np.nan)):
                pts = [(r["shoulder_x"], r["shoulder_y"]),
                       (r["elbow_x"], r["elbow_y"]),
                       (r["wrist_x"], r["wrist_y"])]
                kept = not pd.isna(r.get("elbow_angle_filtered", np.nan))
                colour = green if kept else red
                pts = [(int(x), int(y)) for x, y in pts if not pd.isna(x)]
                if len(pts) == 3:
                    cv2.line(frame, pts[0], pts[1], colour, 3)
                    cv2.line(frame, pts[1], pts[2], colour, 3)
                    for p in pts:
                        cv2.circle(frame, p, 6, colour, -1)
                    label = (f"{r['elbow_angle_raw']:.0f} deg" if kept
                             else "below confidence floor")
                    cv2.putText(frame, label, (pts[1][0] + 12, pts[1][1] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.putText(frame, f"frame {idx}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2)
            writer.write(frame)
            idx += 1
    finally:
        cap.release()
        writer.release()
    return out_path


def _plot(ang: pd.DataFrame, fps: float, out_path: str, title: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = ang["frame"] / fps
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(t, ang["elbow_angle_raw"], color=GREY, lw=1, alpha=0.5,
            label="raw")
    ax.plot(t, ang["elbow_angle_smoothed"], color=FOREST, lw=2.2,
            label="filtered and smoothed")
    ax.axhline(180, color=GREY, ls=":", lw=1.2, label="full extension (180 deg)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Elbow angle (degrees)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def analyse(video, arm: str, backend_name: str, progress=None):
    """Run the pipeline and return (overlay video, plot, feature table, notes)."""
    if video is None:
        return None, None, pd.DataFrame(), "Upload a clip first."

    path = video if isinstance(video, str) else video.name
    fps, w, h, _ = _video_meta(path)

    if progress is not None:
        progress(0.1, desc=f"Running {backend_name}...")

    backend = get_backend(backend_name)
    try:
        rows = [fk.as_row(arm) for fk in backend.run(path, arm)]
    finally:
        backend.close()

    if not rows:
        return None, None, pd.DataFrame(), "No frames were read from that file."

    ang = A.compute_angles(pd.DataFrame(rows))

    if progress is not None:
        progress(0.7, desc="Drawing overlay...")

    tmp = Path(tempfile.mkdtemp())
    overlay = _overlay_video(path, ang, str(tmp / "overlay.mp4"), fps, (w, h))
    plot = _plot(ang, fps, str(tmp / "signal.png"),
                 f"{backend_name} — {arm} arm")

    feats = A.features(ang, fps=fps)
    table = pd.DataFrame([
        ("Frames in clip", f"{feats['n_frames']}"),
        ("Frames retained after confidence filter", f"{feats['n_frames_retained']}"),
        ("Retention", f"{feats['retention'] * 100:.0f}%"),
        ("Peak extension", f"{feats['peak_extension_deg']:.1f} deg"),
        ("Minimum angle", f"{feats['min_angle_deg']:.1f} deg"),
        ("Angular range", f"{feats['angular_range_deg']:.1f} deg"),
        ("Fastest rate of change", f"{feats['max_rate_deg_per_s']:.0f} deg/s"),
        ("Frame rate", f"{fps:.0f} fps"),
    ], columns=["Measurement", "Value"])

    # Notes that change with the result, so the interface is honest about the
    # specific clip rather than showing a fixed caveat.
    notes = []
    ret = feats["retention"]
    if ret < 0.5:
        notes.append(
            f"**Low coverage ({ret * 100:.0f}%).** More than half the frames "
            "fell below the confidence floor, so the signal has large gaps. "
            "Treat the features as indicative only. In the thesis dataset this "
            "pattern is what led to a clip being excluded."
        )
    elif ret < 0.8:
        notes.append(
            f"Coverage is {ret * 100:.0f}%. Some frames were rejected; check "
            "the red sections of the overlay to see where."
        )
    peak = feats["peak_extension_deg"]
    if peak is not None and not np.isnan(peak) and peak > 180:
        notes.append(
            f"**Smoothed peak is {peak:.1f} deg, above 180.** This is the "
            "Savitzky-Golay overshoot documented as defect D5. It is reported "
            "rather than clamped, because clamping would conceal it."
        )
    notes.append(
        f"Angular range here is {feats['angular_range_deg']:.1f} deg. Do not "
        "compare that to the 15 deg ICC tolerance: the tolerance applies to "
        "extension between the start of the arm action and release, and this "
        "measurement carries roughly +/-20 deg of error in any case."
    )

    return overlay, plot, table, "\n\n".join(notes)


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #

def build():
    import gradio as gr

    with gr.Blocks(title="Elbow-angle analysis") as demo:
        gr.Markdown("# Elbow-angle analysis from bowling video")
        gr.Markdown(
            "Sanija Abeywickrama · Bachelor of Computing Science (Honours), UTS"
            " · Supervisor: Dr Yu-Kai Wang"
        )
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(label="Bowling clip (trimmed to the delivery)")
                arm = gr.Radio(["right", "left"], value="right",
                               label="Bowling arm",
                               info="The athlete's own arm, not the side of the frame.")
                backend = gr.Radio(["mediapipe", "yolo11pose"], value="mediapipe",
                                   label="Pose estimator",
                                   info="MediaPipe is faster; YOLO11-pose is a "
                                        "different architecture used as a check.")
                run = gr.Button("Analyse", variant="primary")
                gr.Markdown(
                    f"*Frames where any of the three keypoints falls below a "
                    f"confidence of {A.CONF_THRESHOLD} are excluded rather than "
                    f"interpolated, and shown in red on the overlay.*"
                )
            with gr.Column(scale=2):
                video_out = gr.Video(label="Keypoint overlay")
                plot_out = gr.Image(label="Elbow angle over time")
                table_out = gr.Dataframe(label="Movement features",
                                         headers=["Measurement", "Value"])
                notes_out = gr.Markdown(label="Notes")

        def _go(v, a, b, progress=gr.Progress()):
            return analyse(v, a, b, progress)

        run.click(_go, [video_in, arm, backend],
                  [video_out, plot_out, table_out, notes_out])

    return demo


if __name__ == "__main__":
    build().launch()

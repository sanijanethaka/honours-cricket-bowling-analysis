"""
Stages 3-6: confidence filtering, elbow-angle computation, smoothing, features.

Backend-independent by design. Every function here operates on a keypoints
DataFrame with the schema produced by src/backends.py, so swapping the pose
estimator cannot change the downstream analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

#: Confidence floor. Defended in Chapter 3 against the visibility-error analysis
#: in Ruescas-Nicolau et al. (2024).
CONF_THRESHOLD = 0.7
SMOOTH_WINDOW = 9
SMOOTH_POLY = 3

#: ICC limit on elbow extension between the start and finish of the bowling
#: action. Used as a reference scale for measurement disagreement, never as a
#: legality verdict.
ICC_TOLERANCE_DEG = 15.0

KEYPOINT_COLS = [
    "shoulder_x", "shoulder_y", "elbow_x", "elbow_y", "wrist_x", "wrist_y",
]


def elbow_angle(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    """
    Interior angle at the elbow, in degrees. 180 = fully extended.

    Inputs must be in PIXEL coordinates. Evaluating this in normalised
    coordinates distorts the angle by the frame aspect ratio; see the module
    docstring in src/backends.py for the measured magnitude.
    """
    v1 = shoulder - elbow
    v2 = wrist - elbow
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(v1 @ v2 / (n1 * n2), -1.0, 1.0))))


def compute_angles(
    keypoints: pd.DataFrame,
    conf_threshold: float = CONF_THRESHOLD,
    smooth_window: int = SMOOTH_WINDOW,
    smooth_poly: int = SMOOTH_POLY,
) -> pd.DataFrame:
    """Add raw, confidence-filtered and smoothed elbow-angle columns."""
    df = keypoints.copy()

    raw, filtered = [], []
    for _, r in df.iterrows():
        if r[KEYPOINT_COLS].isna().any():
            raw.append(np.nan)
            filtered.append(np.nan)
            continue
        a = elbow_angle(
            np.array([r["shoulder_x"], r["shoulder_y"]]),
            np.array([r["elbow_x"], r["elbow_y"]]),
            np.array([r["wrist_x"], r["wrist_y"]]),
        )
        raw.append(a)
        min_conf = min(r["shoulder_conf"], r["elbow_conf"], r["wrist_conf"])
        filtered.append(a if min_conf >= conf_threshold else np.nan)

    df["elbow_angle_raw"] = raw
    df["elbow_angle_filtered"] = filtered

    series = pd.Series(filtered, index=df.index)
    gaps = series.isna()
    if series.notna().sum() >= smooth_window:
        interp = series.interpolate(limit_direction="both")
        smoothed = pd.Series(
            savgol_filter(interp.values, smooth_window, smooth_poly), index=df.index
        )
        # Never present a smoothed value where no measurement survived filtering.
        smoothed[gaps] = np.nan
    else:
        smoothed = series.copy()
    df["elbow_angle_smoothed"] = smoothed
    return df


def retention(angles: pd.DataFrame) -> float:
    """Fraction of frames where all three keypoints passed the confidence floor."""
    if len(angles) == 0:
        return 0.0
    return float(angles["elbow_angle_filtered"].notna().sum() / len(angles))


def features(angles: pd.DataFrame, fps: float | None = None) -> dict:
    """
    Movement features from the smoothed signal.

    Rate of change is reported per second as well as per frame, because the
    pilot mixes real-time and slow-motion footage: a per-frame velocity is not
    comparable across clips at different capture rates, and reporting only the
    per-frame figure was a latent comparability defect in the original analysis.
    """
    s = angles["elbow_angle_smoothed"].dropna()
    if s.empty:
        return {
            "n_frames": int(len(angles)), "n_frames_retained": 0, "retention": 0.0,
            "peak_extension_deg": np.nan, "min_angle_deg": np.nan,
            "angular_range_deg": np.nan, "max_rate_deg_per_frame": np.nan,
            "max_rate_deg_per_s": np.nan,
        }
    d = s.diff().abs()
    max_per_frame = float(d.max()) if d.notna().any() else np.nan
    return {
        "n_frames": int(len(angles)),
        "n_frames_retained": int(s.size),
        "retention": retention(angles),
        "peak_extension_deg": float(s.max()),
        "min_angle_deg": float(s.min()),
        "angular_range_deg": float(s.max() - s.min()),
        "max_rate_deg_per_frame": max_per_frame,
        "max_rate_deg_per_s": (
            max_per_frame * fps if (fps and not np.isnan(max_per_frame)) else np.nan
        ),
    }


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """
    Frame-wise agreement between two backends on the same clip.

    Compared on the filtered (pre-smoothing) signal so the result reflects the
    estimators, not the smoother. Only frames where BOTH backends produced a
    surviving measurement are compared, and that overlap is reported -- a high
    agreement over a handful of shared frames is not evidence of much.
    """
    j = pd.DataFrame(
        {"a": a.set_index("frame")["elbow_angle_filtered"],
         "b": b.set_index("frame")["elbow_angle_filtered"]}
    ).dropna()
    n_union = len(set(a["frame"]) | set(b["frame"]))
    if j.empty:
        return {
            "n_frames_compared": 0, "overlap_of_clip": 0.0,
            "mean_abs_diff_deg": np.nan, "median_abs_diff_deg": np.nan,
            "p95_abs_diff_deg": np.nan, "max_abs_diff_deg": np.nan,
            "bias_deg": np.nan, "limits_of_agreement_deg": np.nan,
            "frac_within_icc_tolerance": np.nan, "pearson_r": np.nan,
        }
    diff = j["a"] - j["b"]
    return {
        "n_frames_compared": int(len(j)),
        "overlap_of_clip": float(len(j) / n_union) if n_union else 0.0,
        "mean_abs_diff_deg": float(diff.abs().mean()),
        "median_abs_diff_deg": float(diff.abs().median()),
        "p95_abs_diff_deg": float(diff.abs().quantile(0.95)),
        "max_abs_diff_deg": float(diff.abs().max()),
        # Bland-Altman: systematic bias and 95% limits of agreement.
        "bias_deg": float(diff.mean()),
        "limits_of_agreement_deg": float(1.96 * diff.std(ddof=1)) if len(j) > 1 else np.nan,
        "frac_within_icc_tolerance": float((diff.abs() <= ICC_TOLERANCE_DEG).mean()),
        "pearson_r": float(j["a"].corr(j["b"])) if len(j) > 2 else np.nan,
    }
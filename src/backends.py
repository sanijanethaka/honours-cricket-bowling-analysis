"""
Pose-estimation backends.

Each backend exposes the same interface so that Stages 3-6 of the pipeline are
identical regardless of which estimator produced the keypoints. This is what
makes a like-for-like model comparison possible: only Stage 2 changes.

A backend yields, per frame, the shoulder / elbow / wrist of the nominated
bowling arm in PIXEL coordinates, plus a per-keypoint confidence in [0, 1].

Pixel coordinates matter. MediaPipe returns x and y normalised independently by
frame width and height, so an angle computed in that space is distorted by the
frame aspect ratio. On portrait broadcast footage (e.g. 650x1080, aspect 0.60)
this introduces a mean error of ~8.7 deg and a maximum of ~28 deg relative to
the same keypoints evaluated in pixel space -- larger than the ICC 15 deg
tolerance this project uses as its reference threshold. Every backend therefore
returns pixels, and angle computation happens in pixel space only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


@dataclass
class FrameKeypoints:
    """Bowling-arm keypoints for a single frame, in pixel coordinates."""

    frame: int
    shoulder: Optional[np.ndarray]
    elbow: Optional[np.ndarray]
    wrist: Optional[np.ndarray]
    shoulder_conf: float
    elbow_conf: float
    wrist_conf: float
    detected: bool

    @classmethod
    def empty(cls, frame: int) -> "FrameKeypoints":
        return cls(frame, None, None, None, 0.0, 0.0, 0.0, False)

    def as_row(self, arm: str) -> dict:
        def xy(p):
            return (float(p[0]), float(p[1])) if p is not None else (None, None)

        sx, sy = xy(self.shoulder)
        ex, ey = xy(self.elbow)
        wx, wy = xy(self.wrist)
        return {
            "frame": self.frame,
            "bowling_arm": arm,
            "detected": self.detected,
            "shoulder_x": sx, "shoulder_y": sy, "shoulder_conf": self.shoulder_conf,
            "elbow_x": ex, "elbow_y": ey, "elbow_conf": self.elbow_conf,
            "wrist_x": wx, "wrist_y": wy, "wrist_conf": self.wrist_conf,
        }


class PoseBackend:
    """Interface every estimator must satisfy."""

    name = "base"
    #: What the confidence value means for this backend. Recorded in outputs so
    #: the comparison never silently equates two different quantities.
    confidence_semantics = "undefined"

    def run(self, video_path: str, arm: str) -> Iterator[FrameKeypoints]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# MediaPipe / BlazePose
# --------------------------------------------------------------------------- #

class MediaPipeBackend(PoseBackend):
    """MediaPipe Pose (BlazePose). The project's original Stage 2."""

    name = "mediapipe"
    confidence_semantics = (
        "BlazePose per-landmark visibility: model-internal estimate of whether "
        "the point is present and unoccluded in frame (Bazarevsky et al., 2020)"
    )

    def __init__(self, model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        from mediapipe.python.solutions import pose as mp_pose

        self._mp_pose = mp_pose
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _landmarks_for(self, arm: str):
        lm = self._mp_pose.PoseLandmark
        if arm == "left":
            return lm.LEFT_SHOULDER, lm.LEFT_ELBOW, lm.LEFT_WRIST
        if arm == "right":
            return lm.RIGHT_SHOULDER, lm.RIGHT_ELBOW, lm.RIGHT_WRIST
        raise ValueError(f"bowling_arm must be 'left' or 'right', got {arm!r}")

    def run(self, video_path: str, arm: str) -> Iterator[FrameKeypoints]:
        import cv2

        s_lm, e_lm, w_lm = self._landmarks_for(arm)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        scale = np.array([width, height])

        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                res = self._pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks:
                    L = res.pose_landmarks.landmark
                    s, e, w = L[s_lm], L[e_lm], L[w_lm]
                    yield FrameKeypoints(
                        frame=idx,
                        shoulder=np.array([s.x, s.y]) * scale,
                        elbow=np.array([e.x, e.y]) * scale,
                        wrist=np.array([w.x, w.y]) * scale,
                        shoulder_conf=float(s.visibility),
                        elbow_conf=float(e.visibility),
                        wrist_conf=float(w.visibility),
                        detected=True,
                    )
                else:
                    yield FrameKeypoints.empty(idx)
                idx += 1
        finally:
            cap.release()

    def close(self) -> None:
        self._pose.close()


# --------------------------------------------------------------------------- #
# YOLO11-pose
# --------------------------------------------------------------------------- #

class YoloPoseBackend(PoseBackend):
    """
    YOLO11-pose (Ultralytics), COCO-17 keypoint topology.

    Contrasts with MediaPipe on three axes the comparison is designed to expose:
    top-down detection rather than a single-person detector-tracker, a
    multi-person output that requires explicit subject selection, and a
    keypoint confidence that is a heatmap score rather than an occlusion
    estimate.
    """

    name = "yolo11pose"
    confidence_semantics = (
        "YOLO11-pose per-keypoint score: heatmap/localisation confidence. NOT an "
        "occlusion estimate -- it is not interchangeable with BlazePose visibility"
    )

    # COCO-17 indices
    _IDX = {
        "left": (5, 7, 9),
        "right": (6, 8, 10),
    }

    def __init__(self, weights: str = "yolo11n-pose.pt", imgsz: int = 640):
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._imgsz = imgsz
        self.weights = weights

    def _select_subject(self, result):
        """
        Pick the bowler when several people are detected.

        Heuristic: largest person box by area, which on side-on broadcast
        framing is the bowler in the foreground. Recorded explicitly because
        subject ambiguity is one of this project's documented exclusion
        criteria -- a clip where this heuristic is doing heavy lifting should be
        excluded, not silently accepted.
        """
        if result.boxes is None or len(result.boxes) == 0:
            return None
        areas = [
            float((b[2] - b[0]) * (b[3] - b[1]))
            for b in result.boxes.xyxy.cpu().numpy()
        ]
        return int(np.argmax(areas))

    def run(self, video_path: str, arm: str) -> Iterator[FrameKeypoints]:
        import cv2

        if arm not in self._IDX:
            raise ValueError(f"bowling_arm must be 'left' or 'right', got {arm!r}")
        s_i, e_i, w_i = self._IDX[arm]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(video_path)

        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                res = self._model.predict(
                    frame, imgsz=self._imgsz, verbose=False, device="cpu"
                )[0]
                person = self._select_subject(res)
                kp = res.keypoints
                if person is None or kp is None or kp.xy is None or len(kp.xy) == 0:
                    yield FrameKeypoints.empty(idx)
                else:
                    xy = kp.xy[person].cpu().numpy()
                    conf = (
                        kp.conf[person].cpu().numpy()
                        if kp.conf is not None
                        else np.ones(len(xy))
                    )
                    yield FrameKeypoints(
                        frame=idx,
                        shoulder=xy[s_i], elbow=xy[e_i], wrist=xy[w_i],
                        shoulder_conf=float(conf[s_i]),
                        elbow_conf=float(conf[e_i]),
                        wrist_conf=float(conf[w_i]),
                        detected=True,
                    )
                idx += 1
        finally:
            cap.release()


BACKENDS = {
    "mediapipe": MediaPipeBackend,
    "yolo11pose": YoloPoseBackend,
}


def get_backend(name: str, **kwargs) -> PoseBackend:
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)

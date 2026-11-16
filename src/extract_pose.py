import cv2
import pandas as pd
import os
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# ----------------------------
# SETTINGS
# ----------------------------
VIDEO_PATH = "Data/Clips/MitchellStarc.mp4"
OUTPUT_CSV = "Outputs/starc_delivery_keypoints.csv"
OUTPUT_DEBUG_VIDEO = "Outputs/starc_delivery_debug.mp4"
BOWLING_ARM = "right"

# ----------------------------
# MEDIAPIPE SETUP
# ----------------------------
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Map BOWLING_ARM to the correct MediaPipe landmarks
if BOWLING_ARM == "left":
    SHOULDER_LM = mp_pose.PoseLandmark.LEFT_SHOULDER
    ELBOW_LM    = mp_pose.PoseLandmark.LEFT_ELBOW
    WRIST_LM    = mp_pose.PoseLandmark.LEFT_WRIST
elif BOWLING_ARM == "right":
    SHOULDER_LM = mp_pose.PoseLandmark.RIGHT_SHOULDER
    ELBOW_LM    = mp_pose.PoseLandmark.RIGHT_ELBOW
    WRIST_LM    = mp_pose.PoseLandmark.RIGHT_WRIST
else:
    raise ValueError(f"BOWLING_ARM must be 'left' or 'right', got: {BOWLING_ARM}")

# ----------------------------
# OPEN VIDEO
# ----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs("Outputs", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_DEBUG_VIDEO, fourcc, fps, (width, height))

# ----------------------------
# PROCESS FRAMES
# ----------------------------
rows = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    row = {"frame": frame_idx, "bowling_arm": BOWLING_ARM}

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        shoulder = landmarks[SHOULDER_LM]
        elbow    = landmarks[ELBOW_LM]
        wrist    = landmarks[WRIST_LM]

        row.update({
            "shoulder_x": shoulder.x,
            "shoulder_y": shoulder.y,
            "shoulder_vis": shoulder.visibility,
            "elbow_x": elbow.x,
            "elbow_y": elbow.y,
            "elbow_vis": elbow.visibility,
            "wrist_x": wrist.x,
            "wrist_y": wrist.y,
            "wrist_vis": wrist.visibility,
        })

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
    else:
        row.update({
            "shoulder_x": None,
            "shoulder_y": None,
            "shoulder_vis": None,
            "elbow_x": None,
            "elbow_y": None,
            "elbow_vis": None,
            "wrist_x": None,
            "wrist_y": None,
            "wrist_vis": None,
        })

    rows.append(row)
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
pose.close()

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Done. Saved CSV to: {OUTPUT_CSV}")
print(f"Done. Saved debug video to: {OUTPUT_DEBUG_VIDEO}")
print(f"Bowling arm used: {BOWLING_ARM}")
print(df.head())
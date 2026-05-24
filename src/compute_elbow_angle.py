import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

# ----------------------------
# SETTINGS
# ----------------------------
INPUT_CSV   = "Outputs/starc_delivery_keypoints.csv"
OUTPUT_CSV  = "Outputs/starc_delivery_elbow_angles.csv"
OUTPUT_PLOT = "Outputs/starc_delivery_elbow_angle_plot.png"
BOWLER_NAME = "Mitchell Starc"

VISIBILITY_THRESHOLD = 0.7
SMOOTHING_WINDOW = 9
SMOOTHING_POLY = 3


# ----------------------------
# ANGLE FUNCTION
# ----------------------------
def calculate_angle(shoulder, elbow, wrist):
    """Compute the angle at the elbow joint."""
    v1 = shoulder - elbow
    v2 = wrist - elbow
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return np.nan
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


# ----------------------------
# LOAD KEYPOINTS
# ----------------------------
df = pd.read_csv(INPUT_CSV)

raw_angles = []
filtered_angles = []

for _, row in df.iterrows():
    if (pd.isna(row["shoulder_x"]) or
        pd.isna(row["elbow_x"]) or
        pd.isna(row["wrist_x"])):
        raw_angles.append(np.nan)
        filtered_angles.append(np.nan)
        continue

    shoulder = np.array([row["shoulder_x"], row["shoulder_y"]])
    elbow    = np.array([row["elbow_x"],    row["elbow_y"]])
    wrist    = np.array([row["wrist_x"],    row["wrist_y"]])

    angle = calculate_angle(shoulder, elbow, wrist)
    raw_angles.append(angle)

    min_vis = min(row["shoulder_vis"],
                  row["elbow_vis"],
                  row["wrist_vis"])
    if min_vis < VISIBILITY_THRESHOLD:
        filtered_angles.append(np.nan)
    else:
        filtered_angles.append(angle)

df["elbow_angle_raw"] = raw_angles
df["elbow_angle_filtered"] = filtered_angles


# ----------------------------
# SMOOTHING (Savitzky-Golay)
# ----------------------------
angles_series = pd.Series(filtered_angles)
mask = angles_series.isna()

interp = angles_series.interpolate(limit_direction="both")

if len(interp) >= SMOOTHING_WINDOW:
    smoothed = savgol_filter(interp.values,
                             window_length=SMOOTHING_WINDOW,
                             polyorder=SMOOTHING_POLY)
else:
    smoothed = interp.values

smoothed = pd.Series(smoothed)
smoothed[mask] = np.nan

df["elbow_angle_smoothed"] = smoothed.values


# ----------------------------
# SAVE CSV
# ----------------------------
df.to_csv(OUTPUT_CSV, index=False)


# ----------------------------
# PLOT
# ----------------------------
plt.figure(figsize=(12, 6))
plt.plot(df["frame"], df["elbow_angle_raw"],
         color="lightgray", alpha=0.7, label="Raw")
plt.plot(df["frame"], df["elbow_angle_smoothed"],
         color="steelblue", linewidth=2, label="Smoothed (vis ≥ 0.7)")
plt.xlabel("Frame")
plt.ylabel("Elbow Angle (degrees)")
plt.title(f"{BOWLER_NAME} — Elbow Angle Over Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs("Outputs", exist_ok=True)
plt.savefig(OUTPUT_PLOT, dpi=150)
plt.show()


print(f"Done. CSV saved to: {OUTPUT_CSV}")
print(f"Done. Plot saved to: {OUTPUT_PLOT}")
print(f"Frames kept after filtering: "
      f"{df['elbow_angle_filtered'].notna().sum()} "
      f"/ {len(df)}")
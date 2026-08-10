# Findings from the pipeline rebuild — 10 August 2026

Four defects were found in the preparation-phase pipeline and fixed. Each is
recorded here because each changes numbers that appear in the Autumn submission,
and because three of them are themselves reportable methodological results.

---

## 1. Wrong limb analysed for Mitchell Starc

**What happened.** `src/extract_pose.py` had `BOWLING_ARM = "right"` hardcoded.
Every row of `Outputs/Starc/starc_delivery_keypoints.csv` carries
`bowling_arm=right`. Starc is a left-arm fast bowler, so the published Starc
signal describes his non-bowling arm.

**Why it happened.** The bowling arm was a module-level constant edited by hand
per clip, with no record of what value was used for which clip. Chapter 3
Table 3.1 states Starc is left-arm; the README states "Right (analysed)". Nothing
in the pipeline could detect the contradiction.

**Fix.** The bowling arm now lives in `Data/clips.csv` alongside the clip, and
`src/pipeline.py` validates it on load. The value used is written into every
output row and into `run.json`. Editing code to change a clip parameter is no
longer possible.

**Effect on results.** Starc re-run on the correct (left) arm retains 37% of
frames, not the ~1050 frames previously implied. The re-run signal supersedes
Figure 3.2 in the Autumn submission.

---

## 2. Elbow angle computed in normalised coordinates

**What happened.** MediaPipe returns `x` normalised by frame width and `y`
normalised by frame height, independently. `compute_elbow_angle.py` computed the
angle directly in that space. An angle is only preserved under *uniform*
scaling; normalising each axis by a different constant shears the geometry, so
the reported angle is distorted by the frame aspect ratio.

**Magnitude.** Measured on the existing Cummins keypoints (650 x 1080, aspect
0.60), recomputing in pixel space versus normalised space:

| statistic | value |
|---|---|
| mean absolute difference | **8.7°** |
| maximum absolute difference | **28.2°** |
| peak extension, normalised vs pixel | 179.2° vs 179.5° |

The error is small near full extension (where the vectors are near-collinear and
shear barely rotates them) and largest mid-flexion — which is exactly the region
the angular-range and rate-of-change features are computed over. All clips in the
dataset are portrait-aspect, so none escaped it.

**Why it matters beyond a bugfix.** The thesis uses the ICC 15° limit as its
reference scale for what counts as a meaningful angular difference. A
coordinate-space artefact of up to 28° is nearly twice that tolerance. Any claim
calibrated against the ICC threshold was being made on distorted measurements.

**Fix.** All backends return pixel coordinates (`src/backends.py`), and
`src/angles.py` documents that pixel space is a requirement, not a preference.

---

## 3. Rate of change was not comparable across clips

**What happened.** The rate-of-change feature was defined as maximum
frame-to-frame angular change, in degrees per frame. The pilot deliberately
mixes real-time (30 fps) and slow-motion (60 fps) footage, so a per-frame
velocity measures the capture rate as much as the movement.

**Fix.** `src/angles.py` now reports `max_rate_deg_per_frame` and
`max_rate_deg_per_s`; only the per-second figure is used in cross-clip
comparison. Frame rate is recorded per clip in `run.json`.

---

## 4. Smoothed angles can exceed the anatomical ceiling

**What happened.** `arccos` is bounded to [0°, 180°], but the Savitzky-Golay
filter is a polynomial fit and overshoots at turning points. Starc's smoothed
peak extension came out at **181.6°** (MediaPipe) and **184.4°** (YOLO11-pose) —
values the measurement cannot physically produce.

**Status.** Not yet clamped, deliberately. Clamping to 180° would hide the
overshoot; reporting it documents a real property of the smoother near peak
extension, which is the exact region the peak-extension feature reads. To be
resolved before Chapter 4: either report the overshoot as a known smoothing
artefact with its magnitude, or reduce the window near the signal peak.

---

## 5. The confidence threshold does not transfer between estimators

Not a defect in the old code — a finding produced by the new comparison, and the
most consequential one for the project's method.

The 0.7 threshold is defended in Chapter 3 as a filter on BlazePose's
*visibility* output: a model-internal estimate of whether a point is occluded.
YOLO11-pose's per-keypoint score is a *heatmap localisation confidence*. These
are different quantities, and applying the same numeric cutoff to both is not the
same safeguard.

Consequence, measured:

| clip | MediaPipe retention | YOLO11-pose retention |
|---|---|---|
| cummins | 80% | 98% |
| jadeja | 72% | 99% |
| starc (left arm) | 37% | 98% |
| **murali (excluded)** | **4%** | **64%** |

The Muralitharan clip is the case that matters. It was excluded from the pilot
because MediaPipe's visibility filter correctly rejected a back-view delivery.
Under YOLO11-pose the same clip retains 64% of frames and produces a plausible
peak extension of 178.5° — a confident-looking number from footage in which the
bowling arm faces away from the camera. A pipeline that swapped estimators
without revisiting the threshold would have silently admitted a clip the method
is not valid on.

**Implication for Chapter 3.** The exclusion criteria are backend-specific and
must be re-derived per estimator. This strengthens rather than weakens the
"failure as data" argument: the exclusion register caught this, because the
excluded clip was kept and re-runnable.

---

## 6. Model choice is material at the precision this application needs

Headline result of the comparison. Same clips, same Stages 3–6, only the
estimator swapped, compared on frames where both produced a surviving
measurement:

| clip | shared frames | mean abs diff | max abs diff | bias | within ICC ±15° | r |
|---|---|---|---|---|---|---|
| cummins | 35 | 24.9° | 88.4° | +11.2° | 43% | 0.79 |
| jadeja | 207 | 11.2° | 59.4° | +4.9° | 69% | 0.96 |
| starc | 382 | 9.9° | 48.6° | +1.4° | 84% | 0.79 |
| murali (excluded) | 3 | 88.0° | 136.5° | −88.0° | 0% | −0.60 |

On the three included clips, between 16% and 57% of shared frames disagree by
more than the ICC tolerance, and Cummins carries an 11° systematic bias. The
correlations are high — the two estimators agree on the *shape* of the delivery —
but shape agreement is not the same as agreement on an angle that a 15° rule is
applied to.

**This is the semester's central claim.** For descriptive, shape-level analysis
markerless 2D pose is robust to the choice of estimator. For any claim
calibrated against an absolute angular threshold, the estimator is a free
parameter that changes the answer, and single-model studies of broadcast
biomechanics are under-reporting their uncertainty.

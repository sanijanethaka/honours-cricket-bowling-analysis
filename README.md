# Honours Project — 2D Pose-Based Analysis of Elbow Joint Movement in Cricket Bowling

**Author:** Sanija Abeywickrama
**Student ID:** 14486373
**Institution:** University of Technology Sydney
**Supervisor:** Dr Yu-Kai (YK) Wang
**Subject:** 32931 Technology Research Methods (Autumn 2026)
**Degree:** Bachelor of Computing Science (Honours)

---

## Research Question

Can 2D pose keypoints extracted from cricket bowling videos be used to analyse elbow joint movement patterns and identify potentially risky bowling actions during the delivery phase?

## Project Scope

This project develops a 2D pose-based baseline pipeline for analysing elbow joint movement in cricket bowling deliveries. The broader context considers multiple bowling-related injury risks (shoulder stress, lumbar load, front-leg impact, elbow-related issues), but the implementation focuses on elbow movement as a measurable, interpretable case study.

"Potentially risky movement" is defined as movement patterns (unusually high elbow extension, rapid angular velocity, irregular trajectories) that warrant further biomechanical investigation. The project does not make clinical or diagnostic claims.

## Repository Structure

Honours/
- Data/ — clips, metadata, raw videos (video files gitignored)
- Outputs/ — per-bowler artefacts (keypoints CSV, angles CSV, plot, debug video)
  - Pat Cummins/
  - Mitchell Starc/
  - Ravindra Jadeja/
  - Failed clips/ — documented exclusions
- Papers/ — literature collected for review
- Plan/ — project planning documents
- docs/ — learning journal and notes
- src/ — Python scripts
  - extract_pose.py — MediaPipe pose extraction (generalised left/right arm)
  - compute_elbow_angle.py — elbow angle computation, smoothing, plotting

## Pipeline Overview

1. Input cricket bowling video (trimmed to delivery phase)
2. Extract 2D pose keypoints using MediaPipe (configurable bowling arm)
3. Compute elbow angle frame-by-frame using shoulder-elbow-wrist vectors
4. Apply visibility-based confidence filtering (threshold = 0.7)
5. Apply Savitzky-Golay smoothing (window = 9, polyorder = 3)
6. Plot raw and smoothed elbow-angle trajectories
7. Compare patterns across bowlers and bowling styles

## Pilot Dataset

| Bowler | Style | Bowling Arm |
|---|---|---|
| Pat Cummins | Right-arm fast | Right |
| Mitchell Starc | Left-arm fast | Right (analysed) |
| Ravindra Jadeja | Left-arm orthodox spin | Left |

Documented exclusions: 1 back-view bowler clip (pose detection failed); 1 two-person-in-frame clip (subject ambiguity).

## Dependencies

- python >= 3.10
- opencv-python
- mediapipe
- numpy
- pandas
- matplotlib
- scipy

## Status

This is an in-progress honours preparation project. The implementation phase continues in the following semester.

## Ethical Stance

Source footage is public-domain broadcast material accessed via YouTube. The analysis is technical and exploratory, not clinical. No identifying medical claims are made about any individual bowler. The framing of "potentially risky" reflects biomechanical signal characteristics, not diagnostic conclusions.

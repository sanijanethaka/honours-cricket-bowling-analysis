# What changed on 10 August 2026

The preparation-phase pipeline was two hardcoded scripts, edited by hand per
clip. It is now a registry-driven batch pipeline supporting more than one pose
estimator. The old scripts are preserved unchanged in `src/legacy/` and in git
history; nothing has been deleted.

## Old -> new

| before | after |
|---|---|
| `src/extract_pose.py` (edit constants per clip) | `src/backends.py` + `Data/clips.csv` |
| `src/compute_elbow_angle.py` (edit constants per clip) | `src/angles.py` |
| run by hand, one clip at a time | `python src/pipeline.py --all --backend mediapipe yolo11pose` |
| features described in Ch. 3, computed nowhere | `src/angles.py:features()` -> `Outputs/features.csv` |
| `Outputs/<Bowler Name>/` (inconsistent naming) | `Outputs/runs/<clip_id>/<backend>/` |
| `Outputs/Failed clips/EXCLUSIONS.md` (prose only) | `status`/`exclusion_reason` in `Data/clips.csv`, prose retained |
| no model comparison | `src/analyse.py` -> `Outputs/model_agreement.csv` + figures |

## Why the restructure was necessary, not cosmetic

Two of the four defects in `docs/FINDINGS.md` were only possible because clip
parameters lived in code: the wrong bowling arm for Starc, and the absence of any
record of which value was used for which clip. Moving them into a validated
registry makes that class of error unrepresentable.

## Reproducing the old numbers

`src/legacy/` still runs under `mediapipe==0.10.14`, but note that it computes
angles in normalised coordinates (defect 2 in `docs/FINDINGS.md`) and so will not
reproduce the corrected figures. It is kept for provenance, not for use.

# Penn Action ground-truth validation — setup

## Why this matters more than more cricket clips

The cricket experiment compares MediaPipe against YOLO11-pose. When they disagree
by 25°, nothing in the data says which one is closer to the truth — both are
estimates. Penn Action ships **hand-annotated 2D joint positions for every
frame**, so the same pipeline can be scored against a reference.

That upgrades the semester's central claim from

> the two estimators disagree by more than the ICC tolerance

to

> here is each estimator's absolute elbow-angle error in degrees, and here is how
> that error changes as the camera moves off side-on

The second version is a measurement-validation result. It is also the thing
Chapter 3 currently cites Ruescas-Nicolau et al. (2024) for and does not test.

Penn Action labels each sequence with a coarse viewpoint (`pose`), which makes
the viewpoint question directly answerable. The planar-geometry assumption in
Section 3.5 predicts error should grow off-axis. This measures whether it does.

## What Penn Action contains

- 2,326 video sequences, 15 action classes
- Frame-level annotation: 13 joints (`x`, `y`), `visibility` flags, `bbox`,
  `train`/`test` split, `nframes`, and a coarse `pose` viewpoint label
- Relevant classes: **baseball_pitch** (closest analogue to a cricket delivery),
  **tennis_serve**, **bowling** — plus **golf_swing** and **baseball_swing** kept
  as a non-overhead contrast group

Project page: http://dreamdragon.github.io/PennAction/

## Step 1 — download and extract

Get the dataset from the project page above (frames + labels). Extract so the
layout is exactly:

    Data/PennAction/
        frames/<sequence_id>/000001.jpg ...
        labels/<sequence_id>.mat

`Data/PennAction/` is gitignored — the frames are large and not ours to
redistribute. Cite the dataset paper in Chapter 3 and record the retrieval date
in your logbook.

## Step 2 — verify the joint mapping BEFORE trusting any numbers

```
python src/penn_action.py --check
```

`JOINT_ORDER` in `src/penn_action.py` encodes the order the 13 joint columns are
stored in. It was written from the dataset README, **not** verified against a
downloaded copy. `--check` validates it against body geometry — in image
coordinates the mean y of shoulder should sit above elbow, elbow above wrist, hip
above knee, and so on. If any check FAILS, fix the constant before going further;
otherwise you would be reporting the error of an elbow angle computed from the
wrong joints.

Expected output ends with `layout OK, mapping verified`.

## Step 3 — run the validation

Start small — this runs both estimators over every frame of every sequence:

```
python src/penn_action.py --limit 10
```

Then scale up once it looks sane:

```
python src/penn_action.py --limit 60
```

Rough cost: MediaPipe is fast, YOLO11-pose is roughly 0.3–1 s per frame on CPU.
Penn Action sequences average ~70 frames, so 60 sequences × 2 backends is a
coffee-length run, not an overnight one. Use `--backend mediapipe` alone for a
quick first pass.

## Outputs

    Outputs/penn_action/sequence_scores.csv      one row per sequence per backend
    Outputs/penn_action/summary_by_viewpoint.csv aggregated by backend × viewpoint
    Outputs/penn_action/per_frame.csv            truth vs estimate, every frame
    Outputs/penn_action/error_by_viewpoint.png   error by viewpoint and by class

Key columns in `sequence_scores.csv`:

| column | meaning |
|---|---|
| `mae_deg`, `rmse_deg` | absolute error against human annotation |
| `bias_deg` | signed error — does the estimator systematically over- or under-read? |
| `frac_within_icc` | share of frames within ±15° of truth |
| `coverage` | share of annotated-visible frames the estimator's 0.7 filter kept |
| `viewpoint` | Penn Action's `pose` label — the independent variable |
| `arm_analysed` | which arm was scored, and why it is a heuristic (see below) |

## Two honest caveats

**Arm selection is a heuristic.** Penn Action does not label handedness, so the
code analyses whichever arm is annotated visible in more frames. That is recorded
per sequence in `arm_analysed` so the choice is auditable, but it is a guess, not
a fact about the athlete.

**Coverage is reported separately from error.** Frames the confidence filter
rejected are not counted as errors — they are refusals. An estimator that keeps
40% of frames at 5° error is a different proposition from one that keeps 95% at
12°, and collapsing those into one number would hide the trade-off. Read
`coverage` alongside `mae_deg`.

## What this does not do

Penn Action contains no cricket. It validates the *measurement*, not the cricket
comparison. RO3 — differences between fast and spin bowling styles — still needs
cricket clips, and the dataset is still 3 included clips. Penn Action strengthens
the method; it does not substitute for the domain data.

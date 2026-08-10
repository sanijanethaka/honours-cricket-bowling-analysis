# Clip collection brief — target Thursday 13 August

Your job for the next three days. Everything else is already automated: drop
clips in `Data/Clips/`, add a row to `Data/clips.csv`, run two commands.

## Target: 8–12 new clips

Aim for this spread, because the comparison is per-style and per-arm:

| category | bowling arm | how many | why |
|---|---|---|---|
| Fast | right | 3 | baseline group, currently n=1 (Cummins) |
| Fast | left | 2 | currently n=1 (Starc), and it's slow-motion only |
| Spin (finger/orthodox) | right | 2 | currently n=0 right-arm spin |
| Spin (finger/orthodox) | left | 2 | currently n=1 (Jadeja) |
| Wrist spin (leg-spin) | either | 1–2 | optional; a third style makes the comparison less binary |

Ten clips gets the dataset to 13–14. That is enough for per-style means with a
spread, which is the difference between "here are three signals" and "here is a
comparison".

## Inclusion checklist — check BEFORE you download

The pipeline will reject clips that fail these, so screening upfront saves you
the trim time. From the existing exclusion register, the failures were all
predictable from the thumbnail:

1. **Side-on or semi-side-on camera.** The bowling arm must face the camera at
   release. Back-view and front-on are automatic exclusions — this is what killed
   the Muralitharan clip.
2. **One dominant bowler in frame.** No split-screen, no two-bowler comparisons,
   no coaching overlays with a second figure.
3. **Whole body visible** at least from the delivery stride onward.
4. **No graphics over the bowling arm.** Broadcast score bugs and telestrator
   lines corrupt keypoints.
5. **Named professional bowler in a televised match.** Keeps the ethical stance
   in Section 3.6 intact — no net practice, no amateur footage, no training
   ground material.
6. **Prefer real-time over slow-motion** where you have a choice. You already
   have one slow-motion clip; more of them skews the frame-rate mix.

## Trimming

Trim to the delivery phase: from the start of the bowling-arm action to roughly
half a second after release. Cut the run-up. This is the Stage 1 that the
Cummins v1/v2 comparison justified — the pipeline is not going to find the
delivery for you.

QuickTime Player is enough: `Cmd-T`, drag the handles, `Cmd-S`.

## Naming and registry

Save as `Data/Clips/<clip_id>.mp4` where `clip_id` is lowercase surname, plus a
number if you have more than one clip for that bowler:

```
Data/Clips/bumrah.mp4
Data/Clips/bumrah_2.mp4
Data/Clips/rashid.mp4
```

Then add one row per clip to `Data/clips.csv`:

```csv
clip_id,bowler,style,category,bowling_arm,file,status,exclusion_reason,source_url,retrieved
bumrah,Jasprit Bumrah,Right-arm fast,fast,right,bumrah.mp4,included,,https://youtube.com/watch?v=...,2026-08-12
```

- `category` must be `fast` or `spin` — it drives the comparison grouping.
- `bowling_arm` must be `left` or `right`. **Get this right.** This is the field
  that was wrong for Starc, and it is now the single source of truth.
- `source_url` matters: the transparency commitment in Section 3.6 promises a
  per-clip manifest sufficient for reproduction. Fill it in as you download, not
  later. The four existing rows say `TODO` — please backfill those too if you
  still have the links.
- If a clip fails once you see the pipeline output, set `status` to `excluded`
  and write the reason in `exclusion_reason`. Do not delete it. The exclusion
  register is a documented contribution of this project.

## Then run

```bash
python src/pipeline.py --all --backend mediapipe yolo11pose
python src/analyse.py
```

Roughly 10 s/clip for MediaPipe and 60–90 s/clip for YOLO11-pose on CPU, so a
12-clip dataset is about 20 minutes unattended. Outputs land in
`Outputs/runs/<clip_id>/<backend>/`, with `Outputs/features.csv`,
`Outputs/model_agreement.csv` and `Outputs/figures/` regenerated each time.

Send me the `features.csv` and `model_agreement.csv` when it finishes and I'll
fold the numbers into the deck.

## If you fall short

Six clips is still a real result — it triples the pilot. Prioritise in this
order: right-arm spin (fills the empty cell), right-arm fast (thickens the
baseline), then left-arm anything. Do not pad the count with slow-motion or
marginal camera angles; a clip that gets excluded costs you trim time and buys
one more line in the exclusion register.

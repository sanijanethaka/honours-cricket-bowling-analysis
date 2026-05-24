# Documented Exclusions

This folder contains pipeline outputs from clips that were attempted but excluded from the pilot dataset. Each exclusion is preserved as evidence of the inclusion criteria being applied in practice.

## Inclusion Criteria (from methodology)

A clip is included only if it satisfies all of:
1. Side or semi-side camera view (front- and back-view excluded)
2. Full body visible (head to feet)
3. Single dominant bowler in frame
4. Delivery phase clearly visible
5. Sufficient MediaPipe keypoint visibility (right_shoulder_vis, right_elbow_vis, right_wrist_vis ≥ 0.7 for sufficient frames)

## Exclusion 1 — Back-view bowler (net practice clip)

- Reason: Camera angle (back / back-three-quarter view)
- Failure mode: MediaPipe attempted to localise face keypoints on the rear of the head; possible left/right keypoint swap due to subject orientation
- Lesson: Camera angle must be verified visually before download, not assumed from thumbnail

## Exclusion 2 — Murali clip

- Reason: Low keypoint visibility throughout the delivery phase (back-three-quarter view, net interference)
- Evidence: right_elbow_vis values ranged 0.11–0.30 across the clip; only 3–4 frames passed the 0.7 visibility filter
- Pipeline behaviour: The visibility filter correctly rejected most frames, producing a fragmented smoothed signal rather than a misleading interpolation
- Lesson: This case validates the filter design — the pipeline does not silently produce smooth outputs from unreliable inputs


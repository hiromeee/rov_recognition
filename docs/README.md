# Documentation

Documentation for the underwater marker-recognition system built for the NEDO
Challenge for BLUE ECONOMY 2026 (Division 1: disturbance control of underwater
robots). The core recognition pipeline and its ROS 2 packaging are complete.

## Contents

- [pipeline_overview.md](pipeline_overview.md) — stage-by-stage walkthrough of the
  vision pipeline, including the pinhole geometry, tilt correction and MAD filter.
- [ros2_handover.md](ros2_handover.md) — ROS 2 setup, node design, and the
  hardware bring-up checklist for connecting to the real robot.
- [hsv_tuning_guide.md](hsv_tuning_guide.md) — race-day HSV re-tuning workflow
  using the live tuner (`tools/live_tune_hsv.py`); ~6–10 minutes.
- [evaluation/](evaluation/) — framework for objectively evaluating recognition
  accuracy and latency against real footage.
- `assets/` — demo media used in the project README.
- `260501改定_BlueChallenge競技ルール.pdf` — official competition rules (latest
  confirmed revision).

## On evaluation

When changing the algorithm, follow the procedure in [evaluation/](evaluation/)
to compare latency (P50/P95) and accuracy (false negatives / false positives)
quantitatively before and after the change.

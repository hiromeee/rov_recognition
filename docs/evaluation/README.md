# Recognition Evaluation Framework

A framework for objectively verifying that a change to the recognition algorithm
actually improves accuracy while still meeting the real-time requirement
(≤ 33 ms / frame).

## Approach

- **Data:** real footage placed in `videos_production_environment/`.
- **Goal:** reduce missed markers (false negatives) and false detections (false
  positives) while keeping the per-frame processing time within the 30 FPS budget.

## Metrics

1. **Latency** — per-frame processing time should stay stably **≤ 33 ms** (30 FPS).
2. **Accuracy (recall / precision)** — to be computed automatically once a
   ground-truth annotation set is prepared. For now, runs are inspected visually
   to confirm markers track stably without picking up noise.

## Latency

Per-frame latency is reported by the standalone runner. Run the pipeline on a
recorded video and read the summary it prints:

```bash
uv run python -m vision_pipeline.main --config configs/video_hsv_baseline.yaml
# === Summary ===
# latency_p50_ms=...  latency_p95_ms=...  effective_fps=...
```

## Scale validation (0.5 m marker spacing)

Whether the pinhole depth estimate is correctly scaled is checked against the
known 0.5 m marker spacing from the competition rules. The check is implemented
as [`compute_scale_check()`](../../src/vision_pipeline/scale_validation.py): it
fits a line through the detected markers (SVD), measures the median spacing along
it, and compares it to the expected 0.5 m — frames with fewer than two markers
are skipped.

It is covered by [`tests/test_scale_validation.py`](../../tests/test_scale_validation.py)
and is also surfaced live in the standalone runner's per-frame log (the `scale:`
field) when a `camera` section is present in the config.

> A future `scripts/` runner could batch this across a whole video and write a
> `summary.md` + CSV per run; for now the check runs inline and under test.

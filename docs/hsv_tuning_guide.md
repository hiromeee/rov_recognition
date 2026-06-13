# Race-Day HSV Tuning Guide

> **Time required: ~6–10 minutes**
> **Recommended tool: `tools/live_tune_hsv.py` (live HSV tuner)**

---

## 1. Overview

While watching the ROV's live camera feed, you click on the markers and the tool
computes the HSV range automatically and saves it straight into
`configs/camera_hsv_baseline.yaml`.

```text
[ROV] → /camera_driver/camera_image → [live_tune_hsv] → configs/camera_hsv_baseline.yaml
                                            │
                                            ├ live feed + detection overlay
                                            ├ click to sample marker / background
                                            ├ auto HSV estimate + trackbar fine-tune
                                            └ one-key YAML save
```

---

## 2. Prerequisites

| Item | Check |
|------|-------|
| ROV connection | camera feed flowing on `/camera_driver/camera_image` |
| Comms (Zenoh, etc.) | startup steps completed |
| Environment | WSL Ubuntu (ROS 2 Jazzy) + WSLg (for the OpenCV window) |
| Sanity check | `ros2 topic hz /camera_driver/camera_image` shows ~30 Hz |

---

## 3. Launch

```bash
source /opt/ros/jazzy/setup.bash
cd /mnt/c/Users/htana/github/nedo_recognition
python3 tools/live_tune_hsv.py
```

Options:

```bash
python3 tools/live_tune_hsv.py \
    --topic /camera_driver/camera_image \
    --config configs/camera_hsv_baseline.yaml
```

Two windows open:

- **`live_tune_hsv`** — live feed + mask overlay (translucent green) + detection
  bounding boxes (yellow)
- **`controls`** — 9 trackbars (H/S/V lower & upper, morph, min_area, min_circ)

---

## 4. Tuning steps

### Step 1 — Freeze a good frame

While the feed is flowing, press **SPACE** when the markers are clearly visible.
The HUD switches from `LIVE` (green) to `FROZEN` (red).

### Step 2 — Sample markers (left-click)

On the frozen frame, **left-click near the centre of each marker (sphere)**.

- a green dot appears at each click
- the HSV values of the surrounding 7×7 pixels are collected automatically
- click once per visible marker; avoid blown-out specular highlights

### Step 3 — Sample background (right-click)

**Right-click** 2–3 points on non-marker regions (the blue-green water, pipe,
rope). A red dot appears; these are used as "colours to exclude".

### Step 4 — Auto-estimate HSV

Press **A** to compute a proposed HSV range from the samples; the trackbars update
automatically.

- based on the 5th–95th percentiles of the marker HSV distribution
- regions overlapping the background HSV are avoided automatically
- the V lower bound is auto-adjusted when markers are brighter than the background

### Step 5 — Fine-tune

The detection result (yellow boxes + translucent green mask) updates live on the
frozen frame. **Goal: one box per marker, no false positives.**

| Parameter | Purpose | Guidance |
|-----------|---------|----------|
| `H_lo` / `H_hi` | hue range | narrow it if the mask bleeds into the background |
| `S_lo` | saturation floor | ~0–30 (water cuts saturation hard) |
| `V_lo` | value floor | above the water level (140–180) |
| `morph` | morphology kernel | 3–5; larger to fill holes more aggressively |
| `min_area` | min area (px²) | 100–200; removes small noise |
| `min_circ` | min circularity ×100 | 25–40; removes elongated false positives (pipe, etc.) |

### Step 6 — Verify on another frame

Press **SPACE** to return to live mode and confirm stable detection on a different
moment:

- do the boxes track as the ROV sways?
- any false positives on the pipe/rope?
- is the detection count (`det=N` on the HUD) as expected?

### Step 7 — Save

Press **S** to save into `configs/camera_hsv_baseline.yaml`.

- only the `hsv:` block is replaced
- other sections (source/runtime/camera) and comments are preserved
- the terminal logs `Saved HSV block to ...`

### Step 8 — Quit

Press **ESC**.

---

## 5. Key bindings

| Key | Action |
|-----|--------|
| `SPACE` | freeze / resume frame |
| left-click | sample a marker (only when FROZEN) |
| right-click | sample background (only when FROZEN) |
| `A` | auto-estimate HSV range into the trackbars |
| `C` | clear all click samples (start over) |
| `S` | save to the config file |
| `ESC` | quit |

---

## 6. Race-day flow

```bash
# Terminals 1–3: bring up comms + production nodes
# → make /camera_driver/camera_image available from the ROV

# Terminal 4: HSV tuning (this guide)
source /opt/ros/jazzy/setup.bash
cd /mnt/c/Users/htana/github/nedo_recognition
python3 tools/live_tune_hsv.py
# → SPACE freeze → left-click ×N (markers) → right-click ×N (background)
# → A (auto) → fine-tune → SPACE to verify on another frame → S (save) → ESC

# Then restart recognition_node so it picks up the new hsv: block
```

> **Important:** `recognition_node` reads the config at startup, so it must be
> **restarted** after tuning.

---

## 7. Acceptance criteria

| Check | Expected |
|-------|----------|
| Marker region | mask (translucent green) covers the whole sphere |
| Background (water/pipe/rope) | little or no mask |
| Detection boxes | match the marker count (HUD `det=N`) |
| circularity value | 0.4–0.8 (shown on the HUD) |
| Detection on other frames | stable as the ROV sways |

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| Stuck on `Waiting for frames...` | check topic name (`ros2 topic list`), pass `--topic` |
| OpenCV window won't open | check `echo $DISPLAY`, restart WSL |
| Mask covers the whole screen | H/S/V range too wide; press C and re-sample |
| Markers not detected | lower min_circ / min_area, set morph to 3 |
| Too many false positives | raise min_circ to 30–40, raise V_lo |
| Clicks produce no dots | still in LIVE mode → press SPACE to freeze |
| Save (S) fails | check the config path (`--config`) |

---

## 9. Fallback: manual tuning

If the tool is unavailable, edit `configs/camera_hsv_baseline.yaml` directly.

```yaml
hsv:
  # water cuts saturation hard, so keep the S floor low
  lower: [30, 0, 140]      # H_lo, S_lo, V_lo
  upper: [85, 255, 255]    # H_hi, S_hi, V_hi
  morphology_kernel: 3
  min_contour_area: 100
  min_diameter_px: 10
  min_circularity: 0.25
```

| Colour | Hue (OpenCV: 0–179) |
|--------|---------------------|
| yellow | 20–35 |
| yellow-green | 35–70 |
| green | 70–85 |
| cyan | 85–100 |

**Underwater rules of thumb:**

- saturation drops sharply underwater → **S floor 0–30**
- background water is blue-green around H = 95–110 → keep the marker H range below it
- markers are brighter than the background → **set V floor higher (140–180)**
- false positives on pipe/rope → **min_circularity ≥ 0.25**

Verify after editing:

```bash
# use the live tuner reading the same --config file, or restart the node:
make build && make run
```

---

## 10. Race-day checklist

```text
[ ] ROV connected, ros2 topic hz /camera_driver/camera_image ~30 Hz
[ ] python3 tools/live_tune_hsv.py opens two windows
[ ] SPACE freeze → left-click ×4 markers → right-click ×2–3 background
[ ] A auto-estimates HSV → trackbars update
[ ] all markers get yellow boxes on the frozen frame
[ ] SPACE → verify stable detection on another frame
[ ] S saves → log shows "Saved HSV block to ..."
[ ] restart recognition_node → confirm detections on /recognition_node/debug_image
```

# Pipeline Overview

A stage-by-stage walkthrough of how a raw underwater camera frame becomes a
single vertical-deviation value in metres.

## Full flow

```text
                ┌──────────────────────────────────────────┐
  Camera frame  │  ① Undistort (optional)                  │
  ───────────▶  │     remove barrel lens distortion         │
                └────────────────────┬─────────────────────┘
                                     ▼
                ┌──────────────────────────────────────────┐
                │  ② HSV threshold                          │
                │     keep only the marker colour band       │
                └────────────────────┬─────────────────────┘
                                     ▼
                ┌──────────────────────────────────────────┐
                │  ③ Morphology (open + close)              │
                │     remove specks, close holes             │
                └────────────────────┬─────────────────────┘
                                     ▼
                ┌──────────────────────────────────────────┐
                │  ④ Contour detection                      │
                │     outline every white blob               │
                └────────────────────┬─────────────────────┘
                                     ▼
                ┌──────────────────────────────────────────┐
                │  ⑤ Shape filtering                        │
                │     area · circularity · aspect · border   │
                └────────────────────┬─────────────────────┘
   IMU pitch/roll ───────────────────┤
                                     ▼
                ┌──────────────────────────────────────────┐
                │  ⑥ Pixel → metre                          │
                │     pinhole model + tilt correction        │
                └────────────────────┬─────────────────────┘
                                     ▼
                ┌──────────────────────────────────────────┐
                │  ⑦ MAD outlier filter                     │
                └────────────────────┬─────────────────────┘
                                     ▼
                ┌──────────────────────────────────────────┐
                │  ⑧ Aggregate to one target                │
                └────────────────────┬─────────────────────┘
                                     ▼
                ┌──────────────────────────────────────────┐
                │  ⑨ Output                                 │
                │     Float32 offset [m] + annotated image   │
                └──────────────────────────────────────────┘
```

On the robot, the ROS 2 node feeds the latest IMU-derived pitch/roll into stage ⑥
every frame; see [ros2_handover.md](ros2_handover.md) for the node wiring.

---

## ① Lens distortion correction

**File:** [../src/vision_pipeline/undistort.py](../src/vision_pipeline/undistort.py)

Wide-angle lenses bow straight lines near the frame edges (barrel distortion),
which biases any position measured from those pixels. When a calibration file
(`configs/camera_calibration.yaml`, holding focal length, principal point and
distortion coefficients) is provided, each frame is remapped to a rectilinear
image. The step is optional — it is enabled only when `calibration_file` is set.

## ② HSV thresholding

**File:** [../src/vision_pipeline/detectors/hsv_contour_detector.py](../src/vision_pipeline/detectors/hsv_contour_detector.py)

Colour is described in Hue / Saturation / Value rather than RGB because HSV
separates "what colour" (H) from "how bright/vivid" (S, V), which is far more
stable under the uneven lighting of an underwater scene. A single
`cv2.inRange()` keeps only pixels inside the marker's colour band and produces a
binary mask.

> **Underwater note:** water strips saturation aggressively, so the S lower bound
> is kept low (0–30). This is the single most important tuning insight for the
> environment.

## ③ Morphology

**File:** `hsv_contour_detector.py`

An **opening** (erode→dilate) removes isolated specks from bubbles, suspended
particles and reflections; a **closing** (dilate→erode) fills small holes inside
a marker so it reads as one solid blob. Both use a configurable square kernel.

## ④ Contour detection

**File:** `hsv_contour_detector.py`

`cv2.findContours(..., RETR_EXTERNAL)` returns the outer boundary of every white
blob — the candidate list passed to the shape filter.

## ⑤ Shape filtering

**File:** `hsv_contour_detector.py`

Each candidate must pass four tests, cheapest first:

| Test | Reject when | Why |
|------|-------------|-----|
| **Area** | `area < min_contour_area` | drops dust/reflection specks |
| **Circularity** | `4π·area / perimeter² < min_circularity` | markers are round; rope/pipe is not |
| **Border margin** | blob touches the frame edge | clipped markers give wrong positions |
| **Aspect ratio** | `max(w,h)/min(w,h) > max_aspect_ratio` | rejects thin/elongated shapes |

The marker centre is taken from the contour's **moments** (centroid), not the
bounding-box centre, so a marker fused with its tether string does not drag the
reported centre toward the pipe.

## ⑥ Pixel → metre

**File:** [../src/vision_pipeline/pipeline.py](../src/vision_pipeline/pipeline.py)

**Pinhole model.** With focal lengths `fx, fy` (either measured from calibration
or derived from the configured field of view), a pixel offset `(dx, dy)` from the
image centre maps to a camera-frame ray:

```text
ray = (dx / fx,  dy / fy,  1)
fx  = (width / 2) / tan(FOV_x / 2)        # when no calibration file is used
```

**Depth from marker size.** Because the real marker diameter is known, depth is
recovered from its apparent pixel diameter:

```text
depth = fy · marker_diameter_m / diameter_px
p_cam = ray · depth
```

**Tilt correction.** The vehicle pitches and rolls in the current, so the camera
ray is rotated back into a level frame using IMU-derived angles:

```text
R = Ry(pitch) · Rx(roll)
p_world = R · p_cam      →  (distance_x_m, distance_y_m, distance_z_m)
```

`distance_y_m` is the vertical deviation the competition scores on. On the robot,
pitch/roll arrive every frame from the IMU (quaternion → Euler in
[imu_utils.py](../ros2_ws/src/nedo_vision/nedo_vision/imu_utils.py)) and are
injected via `camera_override`, keeping attitude correction out of the core
algorithm.

## ⑦ MAD outlier filter

**File:** `pipeline.py`

With several markers in view, a bad depth estimate is rejected using the
**Median Absolute Deviation** — robust to outliers in a way the mean and standard
deviation are not:

```text
median = median(depths)
MAD    = median(|depths − median|)
keep d  ⟺  |d − median| ≤ outlier_sigma · 1.4826 · MAD
```

The `1.4826` factor makes the MAD a consistent estimator of the standard
deviation for normally distributed data, so `outlier_sigma` reads as a familiar
"σ" threshold (default 3.0).

## ⑧ Aggregation

**File:** `pipeline.py`

Remaining detections are reduced to one steering target. The strategy is
configurable; the shipped configs use **`nearest_center`** (the marker closest to
the vertical centre-line), which maps directly to the scoring rule. When the key
is absent, the code falls back to `average`.

| Strategy | Picks |
|----------|-------|
| `nearest_center` | marker nearest the vertical centre-line *(used by the shipped configs)* |
| `leftmost` / `rightmost` | extreme marker on one side |
| `largest` | largest area (nearest marker) |
| `average` | centroid of all detections *(code fallback)* |

## ⑨ Output

**Files:** `pipeline.py`, [../src/vision_pipeline/main.py](../src/vision_pipeline/main.py)

- **Annotated image** — bounding boxes, the centroid crosshair, the vertical
  offset line, the aggregated-target marker, and a HUD (frame number, detection
  count, latency, vertical offset in metres). Published continuously as the ROS 2
  debug image and optionally shown in a local window.
- **Vertical offset (ROS 2)** — `std_msgs/Float32` published at the instant a
  marker crosses the vertical centre-line, detected as a sign change in the
  aggregated target's horizontal offset.

---

## Key parameters

See [../configs/video_hsv_baseline.yaml](../configs/video_hsv_baseline.yaml).

```yaml
camera:
  fov_x_deg: 80.0          # horizontal field of view
  fov_y_deg: 64.0          # vertical field of view
  altitude_m: 1.5          # fallback distance when marker size is unknown
  marker_diameter_m: 0.07  # real marker diameter (used for depth)
  marker_spacing_m: 0.50   # known spacing (used for scale validation)

hsv:
  lower: [25, 75, 100]     # HSV lower bound
  upper: [55, 255, 255]    # HSV upper bound
  morphology_kernel: 5
  min_contour_area: 150
  min_diameter_px: 10        # exclude markers that are too small
  # min_circularity: 0.6     # disabled: circularity drops underwater

runtime:
  aggregation_strategy: nearest_center
  outlier_sigma: 3.0
```

## Performance

Detection runs at roughly **7 ms/frame on 1080p** on a laptop CPU (no GPU), well
inside the 33 ms / 30 FPS real-time budget. The heavy lifting is done by OpenCV's
optimised routines (colour conversion, morphology, contours).

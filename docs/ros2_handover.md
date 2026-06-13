# ROS 2 Package — Handover Notes

**Audience:** whoever sets up ROS 2 / brings the node up on the robot.
**Competition:** NEDO Challenge for BLUE ECONOMY 2026, Division 1 (disturbance
control of underwater robots).

---

## 1. Overview

A ROS 2 node that recognises markers in an underwater camera feed and outputs the
robot's vertical deviation in real time.

### Competition rules (relevant part)

- Course: a 10 m vinyl pipe at 5 m depth, with 19 markers at 0.5 m spacing; the
  ROV makes one round trip along the outside.
- **Scoring:** the smaller the average vertical error (in metres) from the
  horizontal centre-line at the moment a marker crosses the vertical centre-line,
  the higher the score.

### Node I/O

| Direction | Topic | Type |
|-----------|-------|------|
| in | `/camera_driver/camera_image` | `sensor_msgs/msg/Image` |
| in | `/mavlink_communicator/current_pose_acc` *(configurable)* | ROV pose/acc (quaternion) |
| out | `recognition_node/difference` | `std_msgs/msg/Float32` |
| out | `recognition_node/debug_image` | `sensor_msgs/msg/Image` |

`debug_image` continuously publishes the annotated feed (bounding boxes + HUD),
viewable in `rqt_image_view`. When `runtime.display: true` (the current default),
the same feed is shown in a local window on the host PC; press `q` to quit.

**Publish timing:** a value is published only on frames where the aggregated
target's `norm_offset_x` (normalised horizontal offset) **crosses zero**. The
value is `distance_y_m` (the physical vertical deviation in metres).

---

## 2. Repository layout

Only `ros2_ws/` is needed for production. It is self-contained: even if split out
into its own repository, a single `colcon build` works.

```text
nedo_recognition/
├── src/vision_pipeline/          # core pipeline (canonical source, dev)
├── configs/                      # parameters (canonical source, dev)
├── tests/                        # unit tests (21 passing)
├── Makefile                      # sync / build / run helpers
│
└── ros2_ws/                      # ★ production ROS 2 workspace (self-contained)
    ├── configs/                  # copied from configs/ by `make sync`
    └── src/
        ├── vision_pipeline_pkg/  # core pipeline (copied by `make sync`)
        │   ├── package.xml
        │   ├── setup.py
        │   └── vision_pipeline/  # ← the source lives here
        │       ├── pipeline.py
        │       ├── config.py
        │       ├── detectors/hsv_contour_detector.py
        │       └── ...
        └── nedo_vision/          # ROS 2 node package
            ├── package.xml
            ├── setup.py
            ├── nedo_vision/
            │   ├── recognition_node.py  # ★ main node
            │   └── imu_utils.py         # ★ quaternion → pitch/roll
            └── launch/
                └── recognition.launch.py
```

### Relationship between `src/` and `ros2_ws/`

`ros2_ws/src/vision_pipeline_pkg/vision_pipeline/` and `ros2_ws/configs/` are
copies of the repository-root `src/vision_pipeline/` and `configs/`, produced by
`make sync`. **Do not edit files inside `ros2_ws/` directly** — edit `src/` and
re-sync.

---

## 3. Setup (Ubuntu 24.04 / ROS 2 Jazzy)

### Prerequisites

```bash
# ROS 2 Jazzy must be installed
sudo apt install ros-jazzy-cv-bridge
```

### Build

Using `ros2_ws/` directly (production / split-out repo):

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install
```

Using `make` from the full repository:

```bash
source /opt/ros/jazzy/setup.bash
cd /path/to/nedo_recognition
make build   # sync (src/ → ros2_ws/) + colcon build
```

### Launch

```bash
# default settings
make run

# override IMU / camera topics
make run-custom IMU=/your/imu/topic CAMERA=/your/camera/topic

# direct ros2 launch
source ros2_ws/install/setup.bash
ros2 launch nedo_vision recognition.launch.py \
    imu_topic:=/your/imu/topic \
    camera_topic:=/camera_driver/camera_image
```

---

## 4. Must-check items when connecting to the real robot

### 4-1. IMU topic name

Confirm the robot's actual topic and pass it as a launch argument.

```bash
ros2 topic list | grep -i pose   # or grep imu
```

### 4-2. IMU coordinate frame (most important)

`quaternion_to_pitch_roll()` in
`ros2_ws/src/nedo_vision/nedo_vision/imu_utils.py` assumes the **ROS 2 standard
ENU / REP-103** frame. If the IMU uses a different convention, fix the axis
assignment in this function.

How to check — tilt the robot by hand while watching the data:

```bash
ros2 topic echo <imu_topic> | grep -A 5 orientation
```

Expected signs:

- `pitch_deg > 0` → robot tilts forward
- `roll_deg > 0` → robot tilts to the right

If the signs are reversed on the real robot, negate the return values in
`imu_utils.py`.

### 4-3. Camera calibration (optional)

`ros2_ws/configs/camera_calibration.yaml` is already generated. To enable it,
uncomment this line in `ros2_ws/configs/camera_hsv_baseline.yaml`:

```yaml
camera:
  # calibration_file: configs/camera_calibration.yaml   ← enable this
```

Enabling distortion correction improves distance-estimation accuracy.

---

## 5. Core design

### Node processing flow

```text
[camera image]
        │
        ▼
  _image_callback()
        │
        ├─ cv_bridge → np.ndarray
        ├─ read latest pitch/roll from IMU (guarded by threading.Lock)
        │      └─ override CameraConfig with pitch/roll (camera_override)
        ├─ VisionPipeline.process_frame(frame, camera_override)
        │      └─ returns (detections, aggregated_target, latency_ms)
        ├─ draw annotation (bbox, green circle, HUD)
        │      ├─ publish to debug_image (always)
        │      └─ cv2.imshow() (only when runtime.display: true, on main thread)
        └─ sign-change check
               prev_norm_offset_x * curr_norm_offset_x < 0
                      │ YES (marker crossed the vertical centre-line)
                      ▼
              publish Float32(distance_y_m) → recognition_node/difference

[imu pose]
        │
        ▼
  _imu_callback()   (runs concurrently: MultiThreadedExecutor)
        │
        └─ quaternion → pitch_deg / roll_deg → stored
```

### Behaviour when a marker is lost

When `aggregated_target is None` (no detection), `_prev_norm_offset_x` is reset to
`None`, and zero-crossing detection restarts from the next frame with a detection.
No spurious publish occurs right after losing a marker.

### Executor

`MultiThreadedExecutor` + two `MutuallyExclusiveCallbackGroup`s:

- image and IMU callbacks run concurrently,
- each callback is not re-entered concurrently (no overlapping image callbacks),
- the shared pitch/roll state is guarded by a `threading.Lock`.

---

## 6. Bring-up checklist

```text
[ ] colcon build succeeds
[ ] recognition_node appears in `ros2 node list`
[ ] recognition_node/difference appears in `ros2 topic list`
[ ] recognition_node/debug_image appears in `ros2 topic list`
[ ] annotated feed is visible in rqt_image_view
[ ] detection logs appear while the camera feed is flowing
[ ] pitch/roll change in the correct direction while IMU data flows (see 4-2)
[ ] difference is published when a marker crosses the vertical centre-line
[ ] the published value [m] roughly matches the visually observed deviation
[ ] processing keeps up at 30 FPS (≤ 33 ms/frame)
```

```bash
ros2 topic hz recognition_node/difference     # processing rate
ros2 topic echo recognition_node/difference   # output value
```

---

## 7. Tuning parameters

HSV detection parameters live in `ros2_ws/configs/camera_hsv_baseline.yaml`.
**Do not modify the algorithm itself** (`ros2_ws/src/vision_pipeline_pkg/...`).

| Symptom | What to adjust |
|---------|----------------|
| No markers detected | `hsv.lower` / `hsv.upper` range |
| Small noise detected | increase `hsv.min_contour_area` / `hsv.min_diameter_px` |
| Markers judged non-circular | lower `hsv.min_circularity` |
| Distance estimate is off | match `camera.fov_x_deg` / `camera.fov_y_deg` to measured values |

When working in the full repository, edit `configs/camera_hsv_baseline.yaml` (the
canonical source) and propagate to `ros2_ws/configs/` with `make sync`.

---

## 8. Known constraints

- **No YOLO** — HSV contour detection only, prioritising speed and reliability.
- **Works without calibration** — omitting `calibration_file` estimates focal
  length from the FOV; less accurate, but functional.
- **`distance_y_m` may be `None`** — when there is no `camera` section, or
  `marker_diameter_m: 0`. In that case `center_offset_y` (pixels) is published
  instead.
- **Past validation material** — archived under `docs/archive/` (gitignored).

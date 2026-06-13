from __future__ import annotations

import math


def quaternion_to_pitch_roll(
    qx: float, qy: float, qz: float, qw: float
) -> tuple[float, float]:
    """クォータニオン (ROS2 標準 ENU/REP-103) から pitch, roll を度で返す。

    Returns:
        (pitch_deg, roll_deg)

    NOTE: IMU の座標系定義（どの軸が pitch/roll に対応するか）は
    実機接続時に要確認。軸の割り当てが異なる場合はここを修正する。
    """
    # Roll (x 軸回り)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll_rad = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y 軸回り)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch_rad = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch_rad = math.asin(sinp)

    return math.degrees(pitch_rad), math.degrees(roll_rad)

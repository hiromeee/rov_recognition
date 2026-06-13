"""imu_utils.quaternion_to_pitch_roll の単体テスト。

各テストケースは「ロボットをこの向きに傾けたとき」という
実機確認の観点でコメントを付けている。実機で符号が逆だった場合は
imu_utils.py の戻り値の符号を修正すること。
"""

import math

from nedo_vision.imu_utils import quaternion_to_pitch_roll


def _euler_to_quat(roll_rad: float, pitch_rad: float, yaw_rad: float) -> tuple:
    """roll-pitch-yaw (extrinsic x-y-z) → quaternion (qx, qy, qz, qw)."""
    cr, sr = math.cos(roll_rad / 2), math.sin(roll_rad / 2)
    cp, sp = math.cos(pitch_rad / 2), math.sin(pitch_rad / 2)
    cy, sy = math.cos(yaw_rad / 2), math.sin(yaw_rad / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def test_identity_no_tilt():
    """水平姿勢 → pitch=0, roll=0。"""
    qx, qy, qz, qw = _euler_to_quat(0.0, 0.0, 0.0)
    pitch, roll = quaternion_to_pitch_roll(qx, qy, qz, qw)
    assert abs(pitch) < 1e-6
    assert abs(roll) < 1e-6


def test_pitch_forward_30deg():
    """ロボットが前方に 30° 傾く → pitch_deg ≈ +30。"""
    qx, qy, qz, qw = _euler_to_quat(0.0, math.radians(30.0), 0.0)
    pitch, roll = quaternion_to_pitch_roll(qx, qy, qz, qw)
    assert abs(pitch - 30.0) < 0.01
    assert abs(roll) < 0.01


def test_pitch_backward_30deg():
    """ロボットが後方に 30° 傾く → pitch_deg ≈ −30。"""
    qx, qy, qz, qw = _euler_to_quat(0.0, math.radians(-30.0), 0.0)
    pitch, roll = quaternion_to_pitch_roll(qx, qy, qz, qw)
    assert abs(pitch - (-30.0)) < 0.01
    assert abs(roll) < 0.01


def test_roll_right_45deg():
    """ロボットが右に 45° 傾く → roll_deg ≈ +45。"""
    qx, qy, qz, qw = _euler_to_quat(math.radians(45.0), 0.0, 0.0)
    pitch, roll = quaternion_to_pitch_roll(qx, qy, qz, qw)
    assert abs(pitch) < 0.01
    assert abs(roll - 45.0) < 0.01


def test_roll_left_45deg():
    """ロボットが左に 45° 傾く → roll_deg ≈ −45。"""
    qx, qy, qz, qw = _euler_to_quat(math.radians(-45.0), 0.0, 0.0)
    pitch, roll = quaternion_to_pitch_roll(qx, qy, qz, qw)
    assert abs(pitch) < 0.01
    assert abs(roll - (-45.0)) < 0.01


def test_gimbal_lock_pitch_90deg():
    """ジンバルロック付近（pitch=90°）でも例外が出ないこと。"""
    qx, qy, qz, qw = _euler_to_quat(0.0, math.radians(90.0), 0.0)
    pitch, roll = quaternion_to_pitch_roll(qx, qy, qz, qw)
    assert abs(pitch - 90.0) < 0.5  # ジンバルロック付近は精度緩め


def test_yaw_only_no_pitch_roll():
    """Yaw のみの回転 → pitch=0, roll=0 のまま。"""
    qx, qy, qz, qw = _euler_to_quat(0.0, 0.0, math.radians(90.0))
    pitch, roll = quaternion_to_pitch_roll(qx, qy, qz, qw)
    assert abs(pitch) < 0.01
    assert abs(roll) < 0.01

"""Convert PiPlus retargeting ``qpos`` files to Holosoma WBT motion NPZ.

The retargeter stores MuJoCo-style qpos as ``[xyz, qwxyz, 22 joints]``.  This
utility can first resample qpos to the WBT control rate, then runs the PiPlus
MuJoCo model for every frame, exports link poses, and finite-differences
positions/orientations to produce the schema consumed by
``holosoma.managers.command.terms.wbt.MotionLoader``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

JOINT_NAMES = [
    # MuJoCo qpos order (the first seven qpos entries are the free root).
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_upper_arm_joint", "r_elbow_joint",
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_upper_arm_joint", "l_elbow_joint",
    "head_yaw_joint", "head_pitch_joint", "r_hip_pitch_joint", "r_hip_roll_joint",
    "r_thigh_joint", "r_calf_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
    "l_hip_pitch_joint", "l_hip_roll_joint", "l_thigh_joint", "l_calf_joint",
    "l_ankle_pitch_joint", "l_ankle_roll_joint",
]

BODY_NAMES = [
    "base_link", "torso_link", "r_shoulder_pitch_link", "r_shoulder_roll_link", "r_upper_arm_link",
    "r_elbow_link", "r_wrist_link", "l_shoulder_pitch_link", "l_shoulder_roll_link", "l_upper_arm_link",
    "l_elbow_link", "l_wrist_link", "head_yaw_link", "head_pitch_link", "camera_link", "r_hip_pitch_link",
    "r_hip_roll_link", "r_thigh_link", "r_calf_link", "r_ankle_pitch_link", "r_ankle_roll_link",
    "l_hip_pitch_link", "l_hip_roll_link", "l_thigh_link", "l_calf_link", "l_ankle_pitch_link",
    "l_ankle_roll_link",
]


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    if len(values) == 1:
        out.fill(0.0)
    else:
        out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
        out[0] = (values[1] - values[0]) / dt
        out[-1] = (values[-1] - values[-2]) / dt
    return out


def _angular_velocity(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from wxyz quaternion sequence."""
    rotations = Rotation.from_quat(quats_wxyz[:, [1, 2, 3, 0]])
    out = np.zeros((len(rotations), 3), dtype=np.float64)
    if len(rotations) == 1:
        return out
    for i in range(len(rotations) - 1):
        out[i] = (rotations[i + 1] * rotations[i].inv()).as_rotvec() / dt
    out[-1] = out[-2]
    return out


def _resample_qpos(qpos: np.ndarray, input_fps: float, output_fps: float) -> np.ndarray:
    """Resample ``[xyz, wxyz, joints]`` qpos while preserving clip duration.

    Translation and joint angles use linear interpolation.  Root orientation
    uses spherical interpolation so the quaternion stays normalized and takes
    the shortest rotation path.  Both endpoints are retained.
    """
    if len(qpos) < 2 or np.isclose(input_fps, output_fps):
        return qpos.copy()
    if input_fps <= 0.0 or output_fps <= 0.0:
        raise ValueError(f"FPS values must be positive, got input={input_fps}, output={output_fps}")

    duration = (len(qpos) - 1) / input_fps
    output_intervals = round(duration * output_fps)
    input_times = np.arange(len(qpos), dtype=np.float64) / input_fps
    output_times = np.linspace(0.0, duration, output_intervals + 1, dtype=np.float64)

    result = np.empty((len(output_times), qpos.shape[1]), dtype=np.float64)
    linear_columns = [*range(3), *range(7, qpos.shape[1])]
    for column in linear_columns:
        result[:, column] = np.interp(output_times, input_times, qpos[:, column])

    root_rotations = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
    result[:, 3:7] = Slerp(input_times, root_rotations)(output_times).as_quat()[:, [3, 0, 1, 2]]
    return result


def convert(input_npz: Path, output_npz: Path, model_xml: Path, output_fps: float | None = None) -> None:
    with np.load(input_npz, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        input_fps = float(np.asarray(data["fps"]).item())
    if qpos.ndim != 2 or qpos.shape[1] != 29:
        raise ValueError(f"Expected qpos shape (T,29), got {qpos.shape}")
    fps = input_fps if output_fps is None else float(output_fps)
    input_frames = len(qpos)
    qpos = _resample_qpos(qpos, input_fps, fps)
    dt = 1.0 / fps

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    if model.nq != 29:
        raise ValueError(f"Model nq={model.nq}, expected 29")
    data = mujoco.MjData(model)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in BODY_NAMES]
    missing = [name for name, idx in zip(BODY_NAMES, body_ids) if idx < 0]
    if missing:
        raise ValueError(f"Model is missing bodies: {missing}")

    body_pos = np.empty((len(qpos), len(BODY_NAMES), 3), dtype=np.float64)
    body_quat = np.empty((len(qpos), len(BODY_NAMES), 4), dtype=np.float64)
    for i, frame in enumerate(qpos):
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        body_pos[i] = data.xpos[body_ids]
        body_quat[i] = data.xquat[body_ids]

    joint_pos = qpos.copy()
    joint_vel = np.zeros((len(qpos), 6 + len(JOINT_NAMES)), dtype=np.float64)
    joint_vel[:, :3] = _finite_difference(joint_pos[:, :3], dt)
    joint_vel[:, 3:6] = _angular_velocity(joint_pos[:, 3:7], dt)
    joint_vel[:, 6:] = _finite_difference(joint_pos[:, 7:], dt)
    body_lin_vel = _finite_difference(body_pos, dt)
    body_ang_vel = np.stack([_angular_velocity(body_quat[:, b], dt) for b in range(len(BODY_NAMES))], axis=1)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        fps=np.asarray(fps, dtype=np.float32),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=body_lin_vel.astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
        joint_names=np.asarray(JOINT_NAMES),
        body_names=np.asarray(BODY_NAMES),
    )
    print(
        f"wrote {output_npz} "
        f"({input_frames} frames at {input_fps:g} Hz -> {len(qpos)} frames at {fps:g} Hz, "
        f"{len(BODY_NAMES)} bodies, {len(JOINT_NAMES)} joints)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_npz", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--model-xml", type=Path, required=True)
    parser.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help="Resample to this FPS before FK and velocity calculation (default: keep input FPS).",
    )
    args = parser.parse_args()
    convert(args.input_npz, args.output_npz, args.model_xml, output_fps=args.output_fps)


if __name__ == "__main__":
    main()

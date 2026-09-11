#!/usr/bin/env python3
"""Retarget an already-retargeted OmniRetarget G1 trajectory to PiPlus.

The input trajectory is the format published by OmniRetarget::

    qpos = [qw, qx, qy, qz, x, y, z, g1_joint_1, ..., g1_joint_29]

This script does not copy G1 joint angles into PiPlus.  It runs forward
kinematics on the G1 model, uses the same 15 body references selected by the
G1 MOCAP mapping plus the ankle-roll and eight foot-corner references, and
solves PiPlus with ``InteractionMeshRetargeter``.  The corner references are
what preserve the source foot plane during the transfer; they are also used
for foot-sticking detection.

Example (one Omni terrain clip)::

    python examples/g1_qpos_retarget.py \
      --g1-qpos-npz /home/sunteng/HT/tool/OmniRetarget_Dataset/robot-terrain/climb_00_z_scale_1.0.npz \
      --terrain-urdf /home/sunteng/HT/tool/OmniRetarget_Dataset/models/terrain/climb_00/multi_boxes_z_scale_1.0.urdf \
      --piplus-urdf-file models/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831/urdf/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831_raw.urdf \
      --piplus-xml-file models/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831/xml/PiPlus_S_12L8A0G2H0W_SSE_ZedMini_260831.xml \
      --save-dir demo_results/piplus_s/climbing/omniretarget

Use ``--max-frames 5`` to validate the model and paths before processing a
complete clip.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import mujoco  # type: ignore[import-not-found]
import numpy as np
import trimesh

# Allow running this file directly from the package checkout.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import (  # noqa: E402
    JOINTS_MAPPINGS,
    MOCAP_DEMO_JOINTS,
)
from holosoma_retargeting.config_types.retargeter import (  # noqa: E402
    FootLockConfig,
    SelfCollisionConfig,
)
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    InteractionMeshRetargeter,
)


# These are exactly the links used by the existing (mocap, g1) mapping in
# config_types/data_type.py.  The values are source G1 body names and the keys
# are the synthetic MOCAP names expected by InteractionMeshRetargeter.
G1_REFERENCE_LINKS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("Spine1", "pelvis_contour_link"),
        ("LeftUpLeg", "left_hip_pitch_link"),
        ("LeftLeg", "left_knee_link"),
        ("LeftFoot", "left_ankle_intermediate_1_link"),
        ("LeftToeBase", "left_ankle_roll_sphere_5_link"),
        ("RightUpLeg", "right_hip_pitch_link"),
        ("RightLeg", "right_knee_link"),
        ("RightFoot", "right_ankle_intermediate_1_link"),
        ("RightToeBase", "right_ankle_roll_sphere_5_link"),
        ("LeftArm", "left_shoulder_roll_link"),
        ("LeftForeArm", "left_elbow_link"),
        ("LeftHandMiddle3", "left_sphere_hand_link"),
        ("RightArm", "right_shoulder_roll_link"),
        ("RightForeArm", "right_elbow_link"),
        ("RightHandMiddle3", "right_sphere_hand_link"),
    ]
)

# G1's four corner spheres per foot are used to determine whether the source
# foot is stationary.  They are deliberately not part of G1_REFERENCE_LINKS.
G1_FOOT_LOCK_LINKS: dict[str, tuple[str, ...]] = {
    "left": (
        "left_ankle_roll_sphere_1_link",
        "left_ankle_roll_sphere_2_link",
        "left_ankle_roll_sphere_3_link",
        "left_ankle_roll_sphere_4_link",
    ),
    "right": (
        "right_ankle_roll_sphere_1_link",
        "right_ankle_roll_sphere_2_link",
        "right_ankle_roll_sphere_3_link",
        "right_ankle_roll_sphere_4_link",
    ),
}

# The original G1->MOCAP map has only an ankle centre and a toe point.  That
# is sufficient for a human skeleton, but it does not fully define a rigid
# foot plane when transferring to a robot with a different ankle-roll offset.
# Reuse the existing MOCAP ``FootMod`` slots for the ankle-roll joint and add
# private synthetic names for the four corner spheres on each foot.  These
# names are appended only by this adapter; normal MOCAP files are unchanged.
G1_FOOT_ROLL_LINKS: dict[str, str] = {
    "left": "left_ankle_roll_link",
    "right": "right_ankle_roll_link",
}
G1_FOOT_CORNER_MOCAP_LINKS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("LeftFootCornerRearOuter", "left_ankle_roll_sphere_1_link"),
        ("LeftFootCornerRearInner", "left_ankle_roll_sphere_2_link"),
        ("LeftFootCornerFrontOuter", "left_ankle_roll_sphere_3_link"),
        ("LeftFootCornerFrontInner", "left_ankle_roll_sphere_4_link"),
        ("RightFootCornerRearInner", "right_ankle_roll_sphere_1_link"),
        ("RightFootCornerRearOuter", "right_ankle_roll_sphere_2_link"),
        ("RightFootCornerFrontInner", "right_ankle_roll_sphere_3_link"),
        ("RightFootCornerFrontOuter", "right_ankle_roll_sphere_4_link"),
    ]
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-qpos-npz", required=True, type=Path)
    parser.add_argument("--terrain-urdf", required=True, type=Path)
    parser.add_argument(
        "--g1-xml-file",
        type=Path,
        default=Path("models/g1/g1_29dof_spherehand.xml"),
        help="MuJoCo G1 model used for FK (default: models/g1/g1_29dof_spherehand.xml)",
    )
    parser.add_argument("--piplus-urdf-file", required=True, type=Path)
    parser.add_argument("--piplus-xml-file", required=True, type=Path)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("demo_results/piplus_s/climbing/omniretarget"),
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help=(
            "Self-contained output directory. The NPZ, terrain URDF, and copied "
            "terrain meshes are written together here."
        ),
    )
    parser.add_argument(
        "--bundle",
        action="store_true",
        help=(
            "Create <save-dir>/<input-stem>_piplus_bundle as a self-contained "
            "output directory."
        ),
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output NPZ basename; defaults to <input-stem>_piplus_original.npz",
    )
    parser.add_argument(
        "--transfer-scale",
        type=float,
        default=None,
        help="Pi/G1 scale. Default is Pi height 0.674 / G1 height 1.32.",
    )
    parser.add_argument("--g1-height", type=float, default=1.32)
    parser.add_argument("--piplus-height", type=float, default=0.674)
    parser.add_argument(
        "--foot-velocity-threshold",
        type=float,
        default=0.01,
        help="Mean XY displacement per frame (m) below which a G1 foot is locked.",
    )
    parser.add_argument("--sample-count", type=int, default=120)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--step-size",
        "--retargeter.step-size",
        dest="step_size",
        type=float,
        default=0.2,
    )
    parser.add_argument("--collision-threshold", type=float, default=0.1)
    parser.add_argument("--penetration-tolerance", type=float, default=1e-3)
    parser.add_argument("--foot-sticking-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--self-collision",
        "--retargeter.self-collision.enable",
        dest="self_collision_enable",
        action="store_true",
        help="Enable self-collision constraints for the configured body pairs.",
    )
    parser.add_argument(
        "--self-collision-pairs",
        "--retargeter.self-collision.pairs",
        dest="self_collision_pair_names",
        nargs="+",
        default=[],
        metavar="BODY",
        help="Flat body-name list: body_a body_b [body_c body_d ...].",
    )
    parser.add_argument(
        "--self-collision-tolerance",
        "--retargeter.self-collision.tolerance",
        dest="self_collision_tolerance",
        type=float,
        default=0.02,
    )
    parser.add_argument("--disable-foot-sticking", action="store_true")
    parser.add_argument("--disable-joint-limits", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if len(args.self_collision_pair_names) % 2:
        parser.error(
            "--self-collision-pairs/--retargeter.self-collision.pairs "
            "requires an even number of body names"
        )
    if args.self_collision_enable and not args.self_collision_pair_names:
        parser.error("self-collision was enabled but no body pairs were provided")
    return args


def _resolve_existing(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _load_omni_qpos(path: Path, max_frames: int | None) -> tuple[np.ndarray, float]:
    path = _resolve_existing(path, "Omni G1 qpos NPZ")
    with np.load(path, allow_pickle=False) as data:
        if "qpos" not in data:
            raise ValueError(f"{path}: expected a qpos key")
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        fps = float(np.asarray(data["fps"] if "fps" in data else 30.0).item())
    if qpos.ndim != 2 or qpos.shape[1] not in (36, 43):
        raise ValueError(f"{path}: expected qpos shape (T,36) or (T,43), got {qpos.shape}")
    # Robot-terrain clips are 36D.  If an object tail is present, preserve only
    # the robot part because this adapter uses a static terrain URDF.
    qpos = qpos[:, :36]
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("--max-frames must be positive")
        qpos = qpos[:max_frames]
    if len(qpos) == 0:
        raise ValueError(f"{path}: qpos contains no frames")
    quat_norm = np.linalg.norm(qpos[:, :4], axis=1)
    if np.any(quat_norm < 1e-8):
        raise ValueError(f"{path}: qpos contains a zero-norm root quaternion")
    qpos[:, :4] /= quat_norm[:, None]
    return qpos, fps


def _omni_to_mujoco_qpos(omni_qpos: np.ndarray) -> np.ndarray:
    """Convert Omni [quat, position, joints] to MuJoCo [position, quat, joints]."""
    out = np.empty_like(omni_qpos)
    out[:, :3] = omni_qpos[:, 4:7]
    out[:, 3:7] = omni_qpos[:, :4]
    out[:, 7:] = omni_qpos[:, 7:36]
    return out


def _fk_g1(qpos_mj: np.ndarray, g1_xml: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    xml = _resolve_existing(g1_xml, "G1 MuJoCo XML")
    model = mujoco.MjModel.from_xml_path(str(xml))
    if model.nq != 36:
        raise ValueError(f"{xml}: expected G1 nq=36, got nq={model.nq}")
    body_ids: dict[str, int] = {}
    all_names = set(G1_REFERENCE_LINKS.values())
    for names in G1_FOOT_LOCK_LINKS.values():
        all_names.update(names)
    all_names.update(G1_FOOT_ROLL_LINKS.values())
    all_names.update(G1_FOOT_CORNER_MOCAP_LINKS.values())
    for name in all_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"{xml}: source G1 body not found: {name}")
        body_ids[name] = int(body_id)

    data = mujoco.MjData(model)
    marker_positions = {name: np.empty((len(qpos_mj), 3), dtype=np.float64) for name in body_ids}
    for frame, q in enumerate(qpos_mj):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for name, body_id in body_ids.items():
            marker_positions[name][frame] = data.xpos[body_id]
    return qpos_mj, marker_positions


def _make_synthetic_mocap(
    marker_positions: dict[str, np.ndarray],
    scale: float,
) -> np.ndarray:
    """Pack G1 FK markers into the MOCAP layout plus private foot points."""
    frames = next(iter(marker_positions.values())).shape[0]
    human = np.zeros((frames, len(MOCAP_DEMO_JOINTS), 3), dtype=np.float64)
    for mocap_name, g1_name in G1_REFERENCE_LINKS.items():
        human[:, MOCAP_DEMO_JOINTS.index(mocap_name)] = marker_positions[g1_name] * scale

    # FootMod is used by this adapter as an ankle-roll reference.  The eight
    # appended corner points preserve the source G1 foot plane and are mapped
    # to PiPlus's corresponding corner spheres below.
    human[:, MOCAP_DEMO_JOINTS.index("Hips")] = marker_positions["pelvis_contour_link"] * scale
    human[:, MOCAP_DEMO_JOINTS.index("LeftFootMod")] = marker_positions[
        G1_FOOT_ROLL_LINKS["left"]
    ] * scale
    human[:, MOCAP_DEMO_JOINTS.index("RightFootMod")] = marker_positions[
        G1_FOOT_ROLL_LINKS["right"]
    ] * scale
    corner_names = list(G1_FOOT_CORNER_MOCAP_LINKS)
    corners = np.zeros((frames, len(corner_names), 3), dtype=np.float64)
    for index, mocap_name in enumerate(corner_names):
        corners[:, index] = marker_positions[G1_FOOT_CORNER_MOCAP_LINKS[mocap_name]] * scale
    return np.concatenate([human, corners], axis=1)


def _foot_sticking_sequence(
    marker_positions: dict[str, np.ndarray],
    velocity_threshold: float,
) -> list[dict[str, bool]]:
    """Detect foot locks from all four G1 corner spheres, not just one toe."""
    left = np.stack([marker_positions[n] for n in G1_FOOT_LOCK_LINKS["left"]], axis=1)
    right = np.stack([marker_positions[n] for n in G1_FOOT_LOCK_LINKS["right"]], axis=1)
    left_vel = np.linalg.norm(np.diff(left[:, :, :2], axis=0), axis=2).mean(axis=1)
    right_vel = np.linalg.norm(np.diff(right[:, :, :2], axis=0), axis=2).mean(axis=1)
    left_vel = np.concatenate([[velocity_threshold + 1.0], left_vel])
    right_vel = np.concatenate([[velocity_threshold + 1.0], right_vel])
    return [
        {
            "L_Toe": bool(left_vel[i] <= velocity_threshold),
            "R_Toe": bool(right_vel[i] <= velocity_threshold),
        }
        for i in range(len(left_vel))
    ]


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            return trimesh.util.concatenate(meshes)
    raise ValueError(f"could not load terrain mesh: {path}")


def _terrain_parts(terrain_urdf: Path, transfer_scale: float) -> list[tuple[str, Path, np.ndarray]]:
    """Read the individual convex mesh parts from an Omni terrain URDF."""
    terrain_urdf = _resolve_existing(terrain_urdf, "Omni terrain URDF")
    root = ET.parse(terrain_urdf).getroot()
    parts: list[tuple[str, Path, np.ndarray]] = []
    for index, link in enumerate(root.findall("link")):
        mesh_elem = link.find("./visual/geometry/mesh")
        if mesh_elem is None:
            mesh_elem = link.find("./collision/geometry/mesh")
        if mesh_elem is None or not mesh_elem.get("filename"):
            continue
        mesh_path = Path(mesh_elem.get("filename", ""))
        if not mesh_path.is_absolute():
            mesh_path = terrain_urdf.parent / mesh_path
        mesh_path = _resolve_existing(mesh_path, "terrain mesh")
        raw_scale = np.fromstring(mesh_elem.get("scale", "1 1 1"), sep=" ", dtype=float)
        if raw_scale.size == 1:
            raw_scale = np.repeat(raw_scale, 3)
        if raw_scale.size != 3:
            raise ValueError(f"{terrain_urdf}: invalid mesh scale on link {link.get('name')}")
        parts.append((f"multi_boxes_omniretarget_{index}", mesh_path, raw_scale * transfer_scale))
    if not parts:
        raise ValueError(f"{terrain_urdf}: no mesh links found")
    return parts


def _terrain_points(parts: list[tuple[str, Path, np.ndarray]], sample_count: int) -> np.ndarray:
    meshes: list[trimesh.Trimesh] = []
    for _, path, scale in parts:
        mesh = _load_mesh(path).copy()
        mesh.vertices *= scale[None, :]
        meshes.append(mesh)
    combined = trimesh.util.concatenate(meshes)
    count = max(16, int(sample_count))
    if len(combined.faces):
        points, _ = trimesh.sample.sample_surface(combined, count=count)
        return np.asarray(points, dtype=np.float64)
    return np.asarray(combined.vertices, dtype=np.float64)


def _make_terrain_urdf(
    parts: list[tuple[str, Path, np.ndarray]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['<?xml version="1.0"?>', '<robot name="omniretarget_terrain">', '  <link name="omniretarget_terrain">']
    for _, mesh_path, scale in parts:
        # Meshes are copied next to this URDF by ``_copy_terrain_meshes``;
        # retain relative filenames so the result can be moved as a folder.
        filename = html.escape(os.path.relpath(mesh_path, output_path.parent), quote=True)
        scale_text = " ".join(f"{value:.10g}" for value in scale)
        lines.append(
            f'    <visual><geometry><mesh filename="{filename}" scale="{scale_text}"/></geometry></visual>'
        )
        lines.append(
            f'    <collision><geometry><mesh filename="{filename}" scale="{scale_text}"/></geometry></collision>'
        )
    lines += ["  </link>", "</robot>", ""]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _copy_terrain_meshes(
    parts: list[tuple[str, Path, np.ndarray]],
    bundle_dir: Path,
) -> list[tuple[str, Path, np.ndarray]]:
    """Copy terrain meshes into ``bundle_dir`` and return portable parts.

    Omni terrain URDFs commonly refer to ``box_models/*.obj``.  Copying each
    source file with a deterministic prefix avoids name collisions while
    preserving the original per-box mesh decomposition used by MuJoCo.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[str, Path, np.ndarray]] = []
    for index, (name, source_path, scale) in enumerate(parts):
        destination = bundle_dir / f"terrain_{index:02d}_{source_path.name}"
        shutil.copy2(source_path, destination)
        copied.append((name, destination, scale))
    return copied


def _make_scene_xml(
    piplus_xml: Path,
    parts: list[tuple[str, Path, np.ndarray]],
    output_path: Path,
) -> Path:
    """Attach each Omni terrain part as a separate MuJoCo mesh geom.

    Keeping each box as its own geom avoids MuJoCo's convex-hull conversion of
    a non-convex combined terrain mesh.  Names contain ``multi_boxes`` so the
    existing object non-penetration filter recognizes them.
    """
    piplus_xml = _resolve_existing(piplus_xml, "PiPlus MuJoCo XML")
    text = piplus_xml.read_text(encoding="utf-8")
    asset_end = text.find("</asset>")
    world_match = re.search(r"(<worldbody(?:\s[^>]*)?>)", text)
    if asset_end < 0 or world_match is None:
        raise ValueError(f"{piplus_xml}: expected <asset> and <worldbody> sections")

    assets: list[str] = []
    geoms: list[str] = []
    for name, mesh_path, scale in parts:
        filename = html.escape(mesh_path.as_posix(), quote=True)
        scale_text = " ".join(f"{value:.10g}" for value in scale)
        assets.append(f'    <mesh name="{name}" file="{filename}" scale="{scale_text}"/>\n')
        geoms.append(
            f'    <geom name="{name}_visual" type="mesh" mesh="{name}" '
            'contype="0" conaffinity="0" rgba="0.45 0.45 0.48 0.55"/>\n'
        )
        geoms.append(
            f'    <geom name="{name}" type="mesh" mesh="{name}" '
            'contype="1" conaffinity="1" rgba="0.45 0.45 0.48 1"/>\n'
        )

    if "omniretarget_terrain" in text:
        raise ValueError(f"refusing to overwrite an existing Omni terrain scene: {piplus_xml}")
    text = text[:asset_end] + "".join(assets) + text[asset_end:]
    # Recompute the worldbody insertion point after asset insertion.
    world_match = re.search(r"(<worldbody(?:\s[^>]*)?>)", text)
    assert world_match is not None
    text = text[: world_match.end()] + "\n" + "".join(geoms) + text[world_match.end() :]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _make_task_constants(
    piplus_urdf: Path,
    piplus_xml: Path,
    scene_xml: Path,
) -> SimpleNamespace:
    robot_config = RobotConfig(
        robot_type="piplus_s",
        robot_dof=22,
        robot_urdf_file=str(piplus_urdf),
        robot_xml_file=str(scene_xml),
    )
    constants = SimpleNamespace()
    for attr in dir(robot_config):
        if attr.isupper() and not attr.startswith("_"):
            setattr(constants, attr, getattr(robot_config, attr))
    constants.ROBOT_XML_FILE = str(scene_xml)
    constants.OBJECT_NAME = "multi_boxes"
    constants.SCENE_XML_FILE = str(scene_xml)
    constants.DEMO_JOINTS = list(MOCAP_DEMO_JOINTS)
    constants.JOINTS_MAPPING = dict(JOINTS_MAPPINGS[("mocap", "piplus_s")])
    constants.DEMO_JOINTS.extend(G1_FOOT_CORNER_MOCAP_LINKS.keys())
    constants.JOINTS_MAPPING.update(
        {
            "LeftFootMod": "l_ankle_roll_link",
            "RightFootMod": "r_ankle_roll_link",
            "LeftFootCornerRearOuter": "l_foot_contact_rear_outer",
            "LeftFootCornerRearInner": "l_foot_contact_rear_inner",
            "LeftFootCornerFrontOuter": "l_foot_contact_front_outer",
            "LeftFootCornerFrontInner": "l_foot_contact_front_inner",
            "RightFootCornerRearInner": "r_foot_contact_rear_inner",
            "RightFootCornerRearOuter": "r_foot_contact_rear_outer",
            "RightFootCornerFrontInner": "r_foot_contact_front_inner",
            "RightFootCornerFrontOuter": "r_foot_contact_front_outer",
        }
    )
    constants.TOE_NAMES = ["LeftToeBase", "RightToeBase"]
    return constants


def _build_retargeter(constants: SimpleNamespace, object_urdf: Path, args: argparse.Namespace):
    pair_names = args.self_collision_pair_names
    self_collision_pairs = [
        (pair_names[index], pair_names[index + 1]) for index in range(0, len(pair_names), 2)
    ]
    return InteractionMeshRetargeter(
        task_constants=constants,
        object_urdf_path=str(object_urdf),
        q_a_init_idx=-7,
        activate_foot_sticking=not args.disable_foot_sticking,
        activate_obj_non_penetration=True,
        activate_joint_limits=not args.disable_joint_limits,
        step_size=args.step_size,
        collision_detection_threshold=args.collision_threshold,
        penetration_tolerance=args.penetration_tolerance,
        foot_sticking_tolerance=args.foot_sticking_tolerance,
        foot_lock=FootLockConfig(),
        self_collision=SelfCollisionConfig(
            enable=args.self_collision_enable,
            pairs=self_collision_pairs,
            tolerance=args.self_collision_tolerance,
        ),
        visualize=args.visualize,
        debug=args.debug,
    )


def main() -> None:
    args = _parse_args()
    if args.bundle and args.bundle_dir is not None:
        raise ValueError("use either --bundle or --bundle-dir, not both")
    qpos_omni, fps = _load_omni_qpos(args.g1_qpos_npz, args.max_frames)
    qpos_g1_mj = _omni_to_mujoco_qpos(qpos_omni)
    _, source_markers = _fk_g1(qpos_g1_mj, args.g1_xml_file)

    transfer_scale = (
        float(args.transfer_scale)
        if args.transfer_scale is not None
        else float(args.piplus_height) / max(float(args.g1_height), 1e-8)
    )
    if transfer_scale <= 0:
        raise ValueError("--transfer-scale must be positive")
    print(f"[g1_qpos_retarget] frames={len(qpos_omni)}, fps={fps:g}, G1->Pi scale={transfer_scale:.6f}")

    # Omni terrain is already expressed in the G1 world frame.  Scale its
    # meshes by the same factor as the G1 marker trajectory.
    parts = _terrain_parts(args.terrain_urdf, transfer_scale)
    object_points = _terrain_points(parts, args.sample_count)
    args.save_dir = args.save_dir.expanduser().resolve()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.bundle_dir is not None:
        output_dir = args.bundle_dir.expanduser().resolve()
    elif args.bundle:
        output_dir = args.save_dir / f"{args.g1_qpos_npz.stem}_piplus_bundle"
    else:
        output_dir = args.save_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    portable_parts = _copy_terrain_meshes(parts, output_dir)
    terrain_urdf_out = output_dir / f"{args.g1_qpos_npz.stem}_terrain_piplus.urdf"
    _make_terrain_urdf(portable_parts, terrain_urdf_out)

    piplus_xml = _resolve_existing(args.piplus_xml_file, "PiPlus MuJoCo XML")
    scene_xml_out = piplus_xml.parent / f"{piplus_xml.stem}_with_omniretarget_{args.g1_qpos_npz.stem}.xml"
    scene_xml = _make_scene_xml(piplus_xml, parts, scene_xml_out)
    piplus_urdf = _resolve_existing(args.piplus_urdf_file, "PiPlus URDF")
    constants = _make_task_constants(piplus_urdf, scene_xml, scene_xml)
    retargeter = _build_retargeter(constants, terrain_urdf_out, args)

    human_joints = _make_synthetic_mocap(source_markers, transfer_scale)
    object_poses = np.tile(np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]), (len(human_joints), 1))
    foot_sticking = _foot_sticking_sequence(source_markers, args.foot_velocity_threshold)

    # Use the scaled source G1 root pose to initialize PiPlus's free joint.
    q_init = np.zeros(7 + constants.ROBOT_DOF, dtype=np.float64)
    q_init[:3] = qpos_g1_mj[0, :3] * transfer_scale
    q_init[3:7] = qpos_g1_mj[0, 3:7]
    object_points_local = np.asarray(object_points, dtype=np.float64)
    output_name = args.output_name or f"{args.g1_qpos_npz.stem}_piplus_original.npz"
    output_path = output_dir / output_name

    retargeter.retarget_motion(
        human_joint_motions=human_joints,
        object_poses=object_poses,
        object_poses_augmented=object_poses,
        object_points_local_demo=object_points_local,
        object_points_local=object_points_local,
        foot_sticking_sequences=foot_sticking,
        q_a_init=q_init,
        q_nominal_list=None,
        original=True,
        dest_res_path=str(output_path),
    )

    # Add convenient metadata without altering the retargeter's standard qpos
    # keys.  The result remains directly consumable by viser_player.py.
    with np.load(output_path, allow_pickle=False) as data:
        saved = {key: data[key] for key in data.files}
    saved.update(
        {
            "fps": np.asarray(fps, dtype=np.float32),
            "source_g1_qpos_npz": np.asarray(str(args.g1_qpos_npz.expanduser().resolve())),
            "terrain_urdf": np.asarray(str(terrain_urdf_out)),
            "terrain_mesh_files": np.asarray(
                [os.path.relpath(path, output_dir) for _, path, _ in portable_parts]
            ),
            "bundle_dir": np.asarray(str(output_dir)),
            "scene_xml": np.asarray(str(scene_xml)),
            "transfer_scale": np.asarray(transfer_scale, dtype=np.float32),
            "reference_link_names": np.asarray(
                list(G1_REFERENCE_LINKS.values())
                + list(G1_FOOT_ROLL_LINKS.values())
                + list(G1_FOOT_CORNER_MOCAP_LINKS.values())
            ),
        }
    )
    np.savez_compressed(output_path, **saved)
    print(f"[g1_qpos_retarget] wrote {output_path}")
    print(f"[g1_qpos_retarget] scene XML: {scene_xml}")
    print(f"[g1_qpos_retarget] terrain URDF: {terrain_urdf_out}")


if __name__ == "__main__":
    main()

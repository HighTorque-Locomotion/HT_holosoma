"""PARC MS motion/terrain adapter for the Holosoma retargeting example.

PARC MS clips are distributed as ``<clip>/<clip>.pkl`` containers.  The
container stores ``motion_data`` (root translation/orientation and local joint
rotations) and ``terrain_data``; the companion ``<clip>_terrain.obj`` is a
portable visualization/collision copy of the terrain.  This module mirrors the
FK and terrain conventions used by ``human-humanoid-tools`` while keeping the
Holosoma example self-contained.

The source PARC rig has 15 bodies and no explicit toe bodies.  Two toe points
are synthesized from the foot body orientation so Holosoma can use its normal
foot-contact and toe mapping logic.
"""

from __future__ import annotations

import html
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

PARC_MS_DEMO_JOINTS: list[str] = [
    "pelvis",
    "torso",
    "head",
    "right_upper_arm",
    "right_lower_arm",
    "right_hand",
    "left_upper_arm",
    "left_lower_arm",
    "left_hand",
    "right_thigh",
    "right_shin",
    "right_foot",
    "left_thigh",
    "left_shin",
    "left_foot",
    # Synthetic contact points; PARC's original 15-body rig terminates at the
    # ankle/foot body and has no toe body.
    "left_toe",
    "right_toe",
]

PARC_MS_PARENTS = np.asarray(
    [-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13],
    dtype=np.int32,
)

# ``humanoid.xml`` body positions in the parent's local frame.  These are the
# same constants used by hhtools' ``parc_ms_skeleton`` adapter.
PARC_MS_LOCAL_TRANSLATION = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.236151],
        [0.0, 0.0, 0.223894],
        [-0.02405, -0.18311, 0.24350],
        [0.0, -0.274788, 0.0],
        [0.0, -0.258947, 0.0],
        [-0.02405, 0.18311, 0.24350],
        [0.0, 0.274788, 0.0],
        [0.0, 0.258947, 0.0],
        [0.0, -0.084887, 0.0],
        [0.0, 0.0, -0.421546],
        [0.0, 0.0, -0.409870],
        [0.0, 0.084887, 0.0],
        [0.0, 0.0, -0.421546],
        [0.0, 0.0, -0.409870],
    ],
    dtype=np.float64,
)

PARC_MS_FOOT_CONTACT_OFFSET_M = 0.05
PARC_MS_TOE_FORWARD_M = 0.12


def parc_ms_clip_paths(data_path: str | Path, task_name: str) -> tuple[Path, Path]:
    """Resolve a PARC MS pickle and its terrain OBJ.

    ``data_path`` may be the selected dataset root or a directory containing a
    single clip.  The expected canonical layout is ``root/task_name/task_name``.
    """

    root = Path(data_path).expanduser().resolve()
    clip_dir = root / task_name
    if not clip_dir.is_dir() and root.name == task_name:
        clip_dir = root
    pkl = clip_dir / f"{task_name}.pkl"
    if not pkl.is_file():
        candidates = sorted(clip_dir.glob("*.pkl")) if clip_dir.is_dir() else []
        if len(candidates) == 1:
            pkl = candidates[0]
        else:
            raise FileNotFoundError(
                f"PARC MS motion pickle not found for {task_name!r} in {clip_dir}; "
                "expected <clip>/<clip>.pkl"
            )
    terrain = clip_dir / f"{pkl.stem}_terrain.obj"
    if not terrain.is_file():
        raise FileNotFoundError(
            f"PARC MS terrain mesh not found: {terrain}; "
            "an incomplete clip cannot be retargeted"
        )
    return pkl, terrain


def _load_blob(container: dict[str, Any], key: str) -> Any:
    raw = container.get(key)
    if raw is None:
        return None
    # Published MS files store inner dictionaries as pickle byte strings.  A
    # few converted copies contain the dictionary directly, so accept both.
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return pickle.loads(raw)
    return raw


def _load_ms_container(path: Path) -> tuple[dict[str, Any], Any, Any]:
    with path.open("rb") as f:
        container = pickle.load(f)
    if not isinstance(container, dict):
        raise ValueError(f"{path}: expected a PARC MS dict container")
    motion = _load_blob(container, "motion_data")
    if not isinstance(motion, dict):
        raise ValueError(f"{path}: missing motion_data blob")
    return motion, _load_blob(container, "terrain_data"), _load_blob(container, "misc_data")


def _normalize_joint_rot(joint_rot: np.ndarray, body_count: int) -> np.ndarray:
    jr = np.asarray(joint_rot, dtype=np.float64)
    if jr.ndim != 3 or jr.shape[-1] != 4:
        raise ValueError(f"joint_rot must have shape (T,J,4), got {jr.shape}")
    expected = body_count - 1
    if jr.shape[1] == expected:
        return jr
    if jr.shape[1] == body_count:
        return jr[:, :expected]
    raise ValueError(
        f"joint_rot has {jr.shape[1]} rows; PARC rig expects {expected} "
        f"or {body_count}"
    )


def _qnormalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, 1e-12)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # xyzw convention
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def _qrotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _qnormalize(q)
    v = np.asarray(v, dtype=np.float64)
    qv = np.zeros_like(q)
    qv[..., :3] = v
    qc = q.copy()
    qc[..., :3] *= -1.0
    return _qmul(_qmul(q, qv), qc)[..., :3]


def load_parc_ms_positions(path: str | Path) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Load a PARC MS pickle and return ``(T,17,3)`` global positions.

    The first 15 rows are the canonical PARC bodies.  Rows 15/16 are synthetic
    left/right toe contact points.  Quaternions are rebuilt internally in xyzw
    order and are not required by Holosoma's position-only interaction mesh.
    """

    pkl_path = Path(path).expanduser().resolve()
    motion, _terrain, misc = _load_ms_container(pkl_path)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = _qnormalize(np.asarray(motion["root_rot"], dtype=np.float64))
    joint_rot = np.asarray(motion["joint_rot"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must have shape (T,3), got {root_pos.shape}")
    if root_rot.shape != (root_pos.shape[0], 4):
        raise ValueError(f"root_rot must have shape ({root_pos.shape[0]},4), got {root_rot.shape}")

    # Match hhtools' import behavior: discard a leading bind-pose frame that
    # is parked at XY=(0,0), since it causes a large artificial initial motion.
    trim = 0
    for i, p in enumerate(root_pos):
        if float(np.linalg.norm(p[:2])) > 1e-3:
            trim = i
            break
    else:
        trim = min(1, max(0, root_pos.shape[0] - 1))
    if trim:
        root_pos = root_pos[trim:]
        root_rot = root_rot[trim:]
        joint_rot = joint_rot[trim:]

    jr = _normalize_joint_rot(joint_rot, len(PARC_MS_PARENTS))
    t = root_pos.shape[0]
    pos = np.zeros((t, len(PARC_MS_PARENTS), 3), dtype=np.float64)
    world_q = np.zeros((t, len(PARC_MS_PARENTS), 4), dtype=np.float64)
    pos[:, 0] = root_pos
    world_q[:, 0] = root_rot
    for j in range(1, len(PARC_MS_PARENTS)):
        parent = int(PARC_MS_PARENTS[j])
        # PARC KinCharModel: parent_rot * bind_rot * local_joint_rot.  The
        # reference humanoid has identity bind rotations.
        world_q[:, j] = _qnormalize(_qmul(world_q[:, parent], jr[:, j - 1]))
        pos[:, j] = pos[:, parent] + _qrotate(world_q[:, parent], PARC_MS_LOCAL_TRANSLATION[j])

    # PARC's foot body origin is the ankle, approximately 5 cm above the sole.
    # Add a forward sole point for Holosoma's toe mapping/contact detector.
    left_foot, right_foot = 14, 11
    left_toe = pos[:, left_foot] + _qrotate(
        world_q[:, left_foot], np.array([PARC_MS_TOE_FORWARD_M, 0.0, -PARC_MS_FOOT_CONTACT_OFFSET_M])
    )
    right_toe = pos[:, right_foot] + _qrotate(
        world_q[:, right_foot], np.array([PARC_MS_TOE_FORWARD_M, 0.0, -PARC_MS_FOOT_CONTACT_OFFSET_M])
    )
    out = np.concatenate([pos, left_toe[:, None, :], right_toe[:, None, :]], axis=1)
    fps = float(np.asarray(motion.get("fps", 30.0)).item())
    meta: dict[str, Any] = {
        "source_format": "parc_ms_pkl",
        "source_pkl": str(pkl_path),
        "trimmed_leading_bind_frames": int(trim),
    }
    if isinstance(misc, dict):
        for key in ("library_folder_label", "mocap_source_take_dir", "source_skeleton_bvh"):
            if misc.get(key) is not None:
                meta[key] = misc[key]
    return out.astype(np.float32), fps, meta


def save_parc_ms_holosoma_npz(
    pkl_path: str | Path,
    output_path: str | Path,
    *,
    human_height: float = 1.70,
) -> Path:
    """Write a portable Holosoma motion NPZ from one PARC MS pickle.

    The file contains the standard ``global_joint_positions`` and ``height``
    keys used by the existing SMPL-X loader, plus source joint names/FPS for
    inspection.  The retargeting CLI uses ``--data-format parc_ms`` directly;
    this export is useful for downstream tools that only understand the generic
    Holosoma NPZ schema.
    """

    positions, fps, meta = load_parc_ms_positions(pkl_path)
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        global_joint_positions=positions,
        height=np.asarray(float(human_height), dtype=np.float32),
        fps=np.asarray(float(fps), dtype=np.float32),
        joint_names=np.asarray(PARC_MS_DEMO_JOINTS),
        source_pkl=np.asarray(meta["source_pkl"]),
    )
    return out


def _as_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            return trimesh.util.concatenate(meshes)
    raise ValueError(f"could not load terrain mesh from {path}")


def create_scaled_terrain_assets(
    terrain_obj: str | Path,
    output_dir: str | Path,
    scale: float,
    *,
    z_scale: float = 1.0,
) -> tuple[Path, Path]:
    """Create a scaled OBJ and a minimal URDF for Viser/object loading."""

    src = Path(terrain_obj).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sx = float(scale)
    stem = f"{src.stem}_scaled_{sx:.6f}_{sx:.6f}_{sx * float(z_scale):.6f}"
    obj_out = out_dir / f"{stem}.obj"
    urdf_out = out_dir / f"{stem}.urdf"
    if not obj_out.is_file():
        mesh = _as_mesh(src).copy()
        mesh.vertices *= np.asarray([sx, sx, sx * float(z_scale)], dtype=np.float64)
        mesh.export(obj_out)
    if not urdf_out.is_file():
        obj_abs = html.escape(obj_out.as_posix(), quote=True)
        urdf_out.write_text(
            '<?xml version="1.0"?>\n'
            '<robot name="parc_ms_terrain">\n'
            '  <link name="parc_ms_terrain">\n'
            f'    <visual><geometry><mesh filename="{obj_abs}"/></geometry></visual>\n'
            f'    <collision><geometry><mesh filename="{obj_abs}"/></geometry></collision>\n'
            "  </link>\n"
            "</robot>\n",
            encoding="utf-8",
        )
    return obj_out, urdf_out


def create_terrain_scene_xml(
    robot_xml: str | Path,
    terrain_obj: str | Path,
    output_xml: str | Path,
    *,
    geom_name: str = "multi_boxes_parc_ms_terrain",
) -> Path:
    """Add a static PARC terrain mesh to an existing MuJoCo robot XML.

    The generated XML is written next to the supplied robot XML by the caller,
    so all original robot mesh paths retain their meaning.  The terrain path is
    absolute to keep per-clip assets independent of the scene location.
    """

    src = Path(robot_xml).expanduser().resolve()
    out = Path(output_xml).expanduser().resolve()
    text = src.read_text(encoding="utf-8")
    terrain_path = html.escape(Path(terrain_obj).resolve().as_posix(), quote=True)
    mesh_line = f'    <mesh name="parc_ms_terrain" file="{terrain_path}"/>\n'
    if 'name="parc_ms_terrain"' not in text:
        match = re.search(r"(<asset(?:\s[^>]*)?>)", text)
        if not match:
            raise ValueError(f"{src}: robot XML has no <asset> section")
        asset_end = text.find("</asset>", match.end())
        if asset_end < 0:
            raise ValueError(f"{src}: malformed <asset> section")
        text = text[:asset_end] + mesh_line + text[asset_end:]
    # MuJoCo uses a convex-hull collision proxy for a non-convex mesh.  A
    # heightfield terrain with a raised platform therefore becomes a large
    # sloped solid, which forces the retargeted feet into the air.  Keep the
    # mesh as a visual-only geom and use one box per heightfield cell for
    # collision.  The OBJ emitted by hhtools is a regular PARC grid, so its
    # unique x/y coordinates recover the exact cell layout without depending
    # on the source pickle being available at scene-load time.
    terrain_mesh = _as_mesh(Path(terrain_obj).expanduser().resolve())
    vertices = np.asarray(terrain_mesh.vertices, dtype=np.float64)
    xs = np.unique(np.round(vertices[:, 0], decimals=7))
    ys = np.unique(np.round(vertices[:, 1], decimals=7))
    boxes: list[str] = []
    if len(xs) >= 2 and len(ys) >= 2:
        heights = np.zeros((len(xs), len(ys)), dtype=np.float64)
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                mask = (np.abs(vertices[:, 0] - x) < 1e-6) & (
                    np.abs(vertices[:, 1] - y) < 1e-6
                )
                if np.any(mask):
                    heights[ix, iy] = float(np.max(vertices[mask, 2]))
        for ix in range(len(xs) - 1):
            for iy in range(len(ys) - 1):
                # Use the maximum corner height so the collision proxy is
                # conservative at a sharp step while remaining non-convex.
                top = max(0.0, float(np.max(heights[ix : ix + 2, iy : iy + 2])))
                thickness = max(top, 0.002)
                cx = 0.5 * (xs[ix] + xs[ix + 1])
                cy = 0.5 * (ys[iy] + ys[iy + 1])
                sx = 0.5 * float(xs[ix + 1] - xs[ix])
                sy = 0.5 * float(ys[iy + 1] - ys[iy])
                boxes.append(
                    f'    <geom name="{geom_name}_cell_{ix}_{iy}" type="box" '
                    f'pos="{cx:.8f} {cy:.8f} {0.5 * thickness:.8f}" '
                    f'size="{sx:.8f} {sy:.8f} {0.5 * thickness:.8f}" '
                    'contype="1" conaffinity="1" rgba="0.45 0.45 0.48 1"/>\n'
                )
    if not boxes:
        raise ValueError(f"{terrain_obj}: expected a regular terrain grid with at least 2x2 vertices")

    geom_line = (
        f'    <geom name="{geom_name}_visual" type="mesh" mesh="parc_ms_terrain" '
        'contype="0" conaffinity="0" rgba="0.45 0.45 0.48 1"/>\n'
        + "".join(boxes)
    )
    if f'name="{geom_name}_visual"' not in text:
        match = re.search(r"(<worldbody(?:\s[^>]*)?>)", text)
        if not match:
            raise ValueError(f"{src}: robot XML has no <worldbody> section")
        text = text[: match.end()] + "\n" + geom_line + text[match.end() :]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


__all__ = [
    "PARC_MS_DEMO_JOINTS",
    "PARC_MS_FOOT_CONTACT_OFFSET_M",
    "create_scaled_terrain_assets",
    "create_terrain_scene_xml",
    "load_parc_ms_positions",
    "parc_ms_clip_paths",
    "save_parc_ms_holosoma_npz",
]

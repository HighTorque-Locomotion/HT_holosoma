#!/usr/bin/env python3
"""Minimal Viser viewer for the raw 53-joint climbing mocap arrays."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
import yourdfpy
from viser.extras import ViserUrdf


# Parent-child edges for MOCAP_DEMO_JOINTS.  The hand chains are included so
# the viewer shows the complete source skeleton, not only retargeted points.
JOINTS = [
    "Hips", "Spine", "Spine1", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3",
    "RightHandRing1", "RightHandRing2", "RightHandRing3",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "LeftFootMod", "RightFootMod",
]


def edges() -> list[tuple[int, int]]:
    parent = {
        "Spine": "Hips", "Spine1": "Spine", "Neck": "Spine1", "Head": "Neck",
        "LeftShoulder": "Neck", "LeftArm": "LeftShoulder", "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
        "RightShoulder": "Neck", "RightArm": "RightShoulder", "RightForeArm": "RightArm", "RightHand": "RightForeArm",
        "LeftUpLeg": "Hips", "LeftLeg": "LeftUpLeg", "LeftFoot": "LeftLeg", "LeftToeBase": "LeftFoot",
        "RightUpLeg": "Hips", "RightLeg": "RightUpLeg", "RightFoot": "RightLeg", "RightToeBase": "RightFoot",
        "LeftFootMod": "LeftFoot", "RightFootMod": "RightFoot",
    }
    for side in ("Left", "Right"):
        hand = f"{side}Hand"
        for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            parent[f"{side}Hand{finger}1"] = hand
            parent[f"{side}Hand{finger}2"] = f"{side}Hand{finger}1"
            parent[f"{side}Hand{finger}3"] = f"{side}Hand{finger}2"
    return [(JOINTS.index(p), JOINTS.index(c)) for c, p in parent.items()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, required=True, help="Raw mocap .npy file")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--terrain-urdf", type=Path, default=None, help="Optional terrain/object URDF")
    args = ap.parse_args()

    motion = np.load(args.file).astype(np.float32)[:: max(1, args.downsample)]
    if motion.ndim != 3 or motion.shape[1:] != (53, 3):
        raise ValueError(f"Expected (T,53,3), got {motion.shape}")

    server = viser.ViserServer()
    server.scene.add_grid("/grid", width=8, height=8, position=(0, 0, 0))

    if args.terrain_urdf is not None:
        if not args.terrain_urdf.exists():
            raise FileNotFoundError(args.terrain_urdf)
        terrain_root = server.scene.add_frame("/terrain", show_axes=False)
        terrain_model = yourdfpy.URDF.load(
            str(args.terrain_urdf), load_meshes=True, build_scene_graph=True
        )
        ViserUrdf(server, urdf_or_path=terrain_model, root_node_name="/terrain")
    sphere = trimesh.primitives.Sphere(radius=0.018)
    points = server.scene.add_batched_meshes_simple(
        "/mocap/joints", vertices=sphere.vertices, faces=sphere.faces,
        batched_positions=motion[0], batched_wxyzs=np.tile([1, 0, 0, 0], (53, 1)),
        batched_colors=(30, 180, 80), opacity=1.0,
    )
    seg = np.asarray([[motion[0, a], motion[0, b]] for a, b in edges()])
    lines = server.scene.add_line_segments("/mocap/skeleton", points=seg, colors=(220, 220, 220), line_width=2.0)

    with server.gui.add_folder("Playback"):
        frame = server.gui.add_slider("Frame", min=0, max=len(motion) - 1, step=1, initial_value=0)
        play = server.gui.add_button("Play / Pause")
        fps = server.gui.add_number("FPS", initial_value=args.fps, min=1, max=240, step=1)
    state = {"playing": False}

    def draw(i: int) -> None:
        points.batched_positions = motion[i]
        lines.points = np.asarray([[motion[i, a], motion[i, b]] for a, b in edges()])

    @frame.on_update
    def _(_) -> None:
        if not state["playing"]:
            draw(int(frame.value))

    @play.on_click
    def _(_) -> None:
        state["playing"] = not state["playing"]

    print(f"Loaded {len(motion)} frames, shape={motion.shape}. Open the Viser URL above.")
    while True:
        if state["playing"]:
            i = (int(frame.value) + 1) % len(motion)
            frame.value = i
            draw(i)
            time.sleep(1.0 / max(1.0, float(fps.value)))
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()

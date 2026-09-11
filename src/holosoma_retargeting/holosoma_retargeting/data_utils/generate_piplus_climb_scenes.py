#!/usr/bin/env python3
"""Generate PiPlus-S + multi-box MuJoCo scenes for the bundled climb demos.

The PiPlus export contains the robot and a floor, while each climb demo owns
its box assets/body include files.  This utility combines those two pieces in
the same layout expected by ``robot_retarget.py``.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


def build_scene(robot_xml: Path, demo_dir: Path, output: Path) -> None:
    content = robot_xml.read_text()

    # The generated scene is stored in demo_dir, while the robot meshes stay
    # in the original PiPlus package.  Keep MuJoCo's meshdir local so the
    # relative paths inside box_assets.xml continue to resolve relative to the
    # demo scene, and make the robot STL references absolute.
    meshdir = robot_xml.parent.parent / "meshes"
    meshdir_from_scene = Path(os.path.relpath(meshdir, demo_dir))
    content = content.replace(
        '<compiler angle="radian" meshdir="../meshes/"/>',
        '<compiler angle="radian" meshdir="."/>',
    )
    content = re.sub(
        r'file="([A-Za-z0-9_]+\.STL)"',
        lambda match: f'file="{meshdir_from_scene / match.group(1)}"',
        content,
    )

    # Insert the per-sequence box mesh assets in the first <asset> block.
    if 'file="box_assets.xml"' not in content:
        marker = "</asset>"
        pos = content.find(marker)
        if pos < 0:
            raise ValueError(f"No <asset> block found in {robot_xml}")
        content = content[:pos] + '    <include file="box_assets.xml"/>\n' + content[pos:]

    # Insert the box bodies in the first (robot) <worldbody> block.  The
    # exported XML has a second worldbody for the floor; inserting at the
    # first closing tag keeps boxes attached to the world alongside the robot.
    if 'file="box_body.xml"' not in content:
        marker = "</worldbody>"
        pos = content.find(marker)
        if pos < 0:
            raise ValueError(f"No <worldbody> block found in {robot_xml}")
        content = content[:pos] + '    <include file="box_body.xml"/>\n' + content[pos:]

    output.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=Path(
            "/home/sunteng/HT/ht_urdf/ht_urdf/"
            "PiPlus_S_12L8A0G2H0W_Fine/xml/PiPlus_S_12L8A0G2H0W_Soccer_New.xml"
        ),
    )
    parser.add_argument(
        "--climb-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "demo_data" / "climb",
    )
    args = parser.parse_args()

    if not args.robot_xml.exists():
        raise FileNotFoundError(args.robot_xml)

    for demo_dir in sorted(args.climb_root.glob("mocap_climb_seq_*")):
        if not demo_dir.is_dir():
            continue
        output = demo_dir / f"{args.robot_xml.stem}_w_multi_boxes.xml"
        build_scene(args.robot_xml, demo_dir, output)
        print(f"generated: {output}")


if __name__ == "__main__":
    main()

"""Merge multiple OBJ meshes into one OBJ, applying optional uniform scales."""

from __future__ import annotations

import argparse
from pathlib import Path


def merge(inputs: list[Path], output: Path, scale: float) -> None:
    vertices: list[str] = []
    faces: list[str] = []
    offset = 0
    for path in inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v" and len(fields) >= 4:
                vertices.append(f"v {float(fields[1]) * scale:.9g} {float(fields[2]) * scale:.9g} {float(fields[3]) * scale:.9g}")
            elif fields[0] == "f":
                mapped = []
                for token in fields[1:]:
                    parts = token.split("/")
                    mapped.append(str(int(parts[0]) + offset))
                faces.append("f " + " ".join(mapped))
        offset = len(vertices)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("# merged terrain OBJ\n" + "\n".join(vertices + faces) + "\n", encoding="utf-8")
    print(f"wrote {output}: {len(vertices)} vertices, {len(faces)} faces, scale={scale:g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    merge(args.inputs, args.output, args.scale)


if __name__ == "__main__":
    main()

"""Convert a PARC MS ``.pkl`` motion into a Holosoma position NPZ.

Example::

    python -m holosoma_retargeting.data_utils.convert_parc_ms \
      --input /path/to/clip/clip.pkl \
      --output /tmp/clip.npz
"""

from __future__ import annotations

import argparse

from holosoma_retargeting.data_utils.parc_ms import save_parc_ms_holosoma_npz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="PARC MS clip .pkl")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument(
        "--human-height",
        type=float,
        default=1.70,
        help="Canonical source human height in metres (default: 1.70)",
    )
    args = parser.parse_args()
    out = save_parc_ms_holosoma_npz(
        args.input,
        args.output,
        human_height=args.human_height,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

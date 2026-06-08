#!/usr/bin/env python3
"""Create a smaller manifest by taking the first N entries from an existing manifest.

Usage:
  python examples/create_manifest_subset.py --in data/manifest_voicebank.json --out data/manifest_voicebank_10.json --n 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(in_path: str, out_path: str, n: int):
    p_in = Path(in_path)
    p_out = Path(out_path)
    with p_in.open("r", encoding="utf-8") as f:
        items = json.load(f)
    subset = items[:n]
    p_out.parent.mkdir(parents=True, exist_ok=True)
    with p_out.open("w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)
    print(f"Wrote {len(subset)} items to {p_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--n", type=int, default=10)
    args = p.parse_args()
    main(args.in_path, args.out_path, args.n)

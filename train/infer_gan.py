#!/usr/bin/env python3
"""Load a trained GAN checkpoint and run inference on a manifest entry."""
from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
import torch

from segan import Generator


def main(manifest: str, ckpt: str, device: str = "cpu"):
    device = torch.device(device)
    with open(manifest, "r", encoding="utf-8") as f:
        items = json.load(f)
    item = items[0]
    mix_path = Path(item["mixture"])

    mix, sr = sf.read(mix_path)
    if mix.ndim > 1:
        mix = mix.mean(axis=1)

    x = torch.from_numpy(mix).float().unsqueeze(0).unsqueeze(0).to(device)

    G = Generator().to(device)
    state = torch.load(ckpt, map_location=device)
    G.load_state_dict(state["G"])
    G.eval()
    with torch.no_grad():
        out = G(x)
    out = out.squeeze().cpu().numpy()

    # match original length
    out = out[: len(mix)]

    out_path = Path(ckpt).resolve().parent / "enhanced_0001.wav"
    sf.write(out_path, out, sr)
    print(f"Wrote enhanced file to {out_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default=("mps" if torch.backends.mps.is_available() else "cpu"))
    args = p.parse_args()
    main(args.manifest, args.ckpt, args.device)

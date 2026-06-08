#!/usr/bin/env python3
"""Clean exporter for VoiceBank-DEMAND-16k.

Usage:
  python examples/hf_export_voicebank_clean.py --out data/voicebank --limit 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import soundfile as sf
import numpy as np
from datasets import load_dataset


DEFAULT_DATASET = "JacobLinCool/VoiceBank-DEMAND-16k"


def detect_fields(example: dict[str, Any]) -> tuple[str, str] | None:
    candidates = [
        ("noisy", "clean"),
        ("mixture", "clean"),
        ("noisy_speech", "clean_speech"),
        ("noisy_audio", "clean_audio"),
    ]
    for a, b in candidates:
        if a in example and b in example:
            return a, b

    audio_like = [k for k, v in example.items() if isinstance(v, dict) and ("array" in v or "sampling_rate" in v)]
    if len(audio_like) >= 2:
        return audio_like[0], audio_like[1]
    return None


def save_audio_field(field: Any, out_path: Path) -> int:
    # Handle HF datasets audio decoder objects (torchcodec AudioDecoder)
    if hasattr(field, "get_all_samples"):
        samples = field.get_all_samples()
        # preferred path: many implementations expose a .data tensor
        if hasattr(samples, "data"):
            arr = np.asarray(samples.data)
            # samples.data may be shape (1, N) for mono
            if arr.ndim > 1 and arr.shape[0] == 1:
                arr = arr.ravel()
            sr = int(getattr(samples, "sample_rate", getattr(field, "_desired_sample_rate", 16000)))
        else:
            # fallback: iterate frames and concatenate
            frames = []
            for frame in samples:
                try:
                    frames.append(np.asarray(frame).ravel())
                except Exception:
                    # best-effort conversion
                    frames.append(np.asarray([frame], dtype="float32"))
            if len(frames) == 0:
                raise RuntimeError("No audio frames found in samples")
            arr = np.concatenate(frames)
            sr = int(getattr(samples, "sample_rate", getattr(field, "_desired_sample_rate", 16000)))
    elif isinstance(field, dict) and "array" in field:
        arr = np.asarray(field["array"])
        sr = int(field.get("sampling_rate", 16000))
    elif isinstance(field, (list, tuple)):
        arr = np.asarray(field)
        sr = 16000
    elif isinstance(field, np.ndarray):
        arr = field
        sr = 16000
    elif isinstance(field, str):
        data, sr = sf.read(field)
        arr = np.asarray(data)
    else:
        raise RuntimeError("Unsupported audio field format")

    if arr.ndim > 1:
        arr = arr.mean(axis=1)

    sf.write(out_path, arr.astype("float32"), sr)
    return sr


def export_dataset(dataset_id: str, out_dir: Path, limit: int | None = None):
    print(f"Loading dataset '{dataset_id}' (this may download files)")
    ds = load_dataset(dataset_id, split="train")
    print("Dataset loaded; items:", len(ds))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_items = []
    count = 0
    if len(ds) == 0:
        print("Dataset contains no items")
        return
    sample = ds[0]
    fields = detect_fields(sample)
    if fields is None:
        print("Could not detect noisy/clean fields automatically. Example keys:", list(sample.keys()))
        raise SystemExit(1)
    noisy_field, clean_field = fields
    print("Using fields:", noisy_field, clean_field)

    for i, ex in enumerate(ds):
        if limit is not None and count >= limit:
            break
        try:
            noisy = ex[noisy_field]
            clean = ex[clean_field]
        except Exception:
            print(f"Skipping item {i} — missing expected fields")
            continue

        mix_path = out_dir / f"mix_{count+1:06d}.wav"
        clean_path = out_dir / f"clean_{count+1:06d}.wav"
        try:
            save_audio_field(noisy, mix_path)
            save_audio_field(clean, clean_path)
        except Exception as e:
            print(f"Skipping item {i} due to error: {e}")
            continue

        manifest_items.append({"mixture": str(mix_path), "targets": [str(clean_path)]})
        count += 1
        if count % 50 == 0:
            print(f"Exported {count} items")

    manifest_path = out_dir.parent / "manifest_voicebank.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_items, f, indent=2)

    print(f"Exported {count} items to {out_dir}; manifest at {manifest_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    args = p.parse_args()

    export_dataset(args.dataset_id, Path(args.out), args.limit)


if __name__ == "__main__":
    main()

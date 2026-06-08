from __future__ import annotations

from pathlib import Path
import json
from typing import Sequence

import soundfile as sf
import torch
from torch.utils.data import Dataset


class ManifestWaveDataset(Dataset):
    """Dataset that loads waveform pairs from a manifest JSON file.

    Manifest is a list of objects: {"mixture": path, "targets": [clean_path], ...}
    """

    def __init__(self, manifest_path: str | Path, transform=None):
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        mix_path = item["mixture"]
        clean_path = item["targets"][0]

        mix, sr1 = sf.read(mix_path)
        clean, sr2 = sf.read(clean_path)
        if sr1 != sr2:
            raise RuntimeError("sample rates do not match")

        # Make mono and shape (1, T)
        if mix.ndim > 1:
            mix = mix.mean(axis=1)
        if clean.ndim > 1:
            clean = clean.mean(axis=1)

        mix = torch.from_numpy(mix).float().unsqueeze(0)
        clean = torch.from_numpy(clean).float().unsqueeze(0)

        if self.transform is not None:
            mix, clean = self.transform(mix, clean)

        return {"mixture": mix, "clean": clean}

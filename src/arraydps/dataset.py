from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import ArrayDPSIOConfig, SignalFormat
from .io import (
    validate_stft_mixture,
    validate_stft_source,
    validate_waveform_mixture,
    validate_waveform_source,
)

Loader = Callable[[str], Any]


@dataclass(frozen=True, slots=True)
class ArrayDPSManifestEntry:
    """One training example in a manifest-based dataset."""

    mixture: str
    targets: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, config: ArrayDPSIOConfig) -> None:
        if not self.mixture:
            raise ValueError("mixture path is required")
        if len(self.targets) != config.num_sources:
            raise ValueError(
                f"expected {config.num_sources} targets, got {len(self.targets)}"
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ArrayDPSManifestEntry":
        mixture = str(data.get("mixture", "")).strip()
        raw_targets = data.get("targets", [])
        if isinstance(raw_targets, (str, bytes)):
            targets = [str(raw_targets)]
        else:
            targets = [str(item) for item in raw_targets]
        metadata = dict(data.get("metadata", {}))
        return cls(mixture=mixture, targets=targets, metadata=metadata)


def load_manifest(manifest_path: str | Path) -> list[ArrayDPSManifestEntry]:
    """Load a JSON or JSONL manifest file.

    JSON format:
    [
      {"mixture": "mix.wav", "targets": ["s1.wav", "s2.wav"]}
    ]

    JSONL format:
    one JSON object per line with the same keys.
    """

    path = Path(manifest_path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("JSON manifest must contain a list of examples")
        return [ArrayDPSManifestEntry.from_mapping(item) for item in payload]

    entries: list[ArrayDPSManifestEntry] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        entries.append(ArrayDPSManifestEntry.from_mapping(item))
    return entries


class ArrayDPSDataset:
    """Manifest-driven dataset wrapper for ArrayDPS.

    The dataset expects a loader function that can read the files referenced by
    the manifest and return an array-like object with a ``shape`` attribute.
    This keeps the module lightweight while remaining compatible with NumPy,
    PyTorch, or any other array library.
    """

    def __init__(
        self,
        manifest: Sequence[ArrayDPSManifestEntry],
        config: ArrayDPSIOConfig,
        loader: Loader,
    ) -> None:
        config.validate()
        self._manifest = list(manifest)
        self._config = config
        self._loader = loader

    def __len__(self) -> int:
        return len(self._manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self._manifest[index]
        entry.validate(self._config)

        mixture = self._loader(entry.mixture)
        targets = [self._loader(path) for path in entry.targets]

        if self._config.signal_format == SignalFormat.WAVEFORM:
            validate_waveform_mixture(mixture, self._config.num_mics)
            for target in targets:
                validate_waveform_source(target)
        else:
            validate_stft_mixture(mixture, self._config.num_mics)
            for target in targets:
                validate_stft_source(target)

        return {
            "mixture": mixture,
            "targets": targets,
            "metadata": dict(entry.metadata),
        }

    @property
    def config(self) -> ArrayDPSIOConfig:
        return self._config

    @property
    def manifest(self) -> list[ArrayDPSManifestEntry]:
        return list(self._manifest)


def describe_manifest(entry: ArrayDPSManifestEntry) -> dict[str, Any]:
    return {
        "mixture": entry.mixture,
        "targets": list(entry.targets),
        "metadata": dict(entry.metadata),
    }

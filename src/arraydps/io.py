from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

from .config import ArrayDPSIOConfig, SignalFormat


def _shape_of(x: Any) -> tuple[int, ...]:
    shape = getattr(x, "shape", None)
    if shape is None:
        raise TypeError("Input must expose a shape attribute")
    return tuple(int(dim) for dim in shape)


def _is_batch_shape(shape: Sequence[int], expected_channels: int) -> bool:
    return len(shape) == 3 and shape[1] == expected_channels


def _is_unbatched_shape(shape: Sequence[int], expected_channels: int) -> bool:
    return len(shape) == 2 and shape[0] == expected_channels


def validate_waveform_mixture(mixture: Any, num_mics: int) -> tuple[int, ...]:
    """Validate a waveform mixture tensor.

    Accepted shapes:
    - [num_mics, num_samples]
    - [batch, num_mics, num_samples]
    """

    shape = _shape_of(mixture)
    if not (_is_unbatched_shape(shape, num_mics) or _is_batch_shape(shape, num_mics)):
        raise ValueError(
            "Waveform mixture must have shape [num_mics, num_samples] or "
            "[batch, num_mics, num_samples]"
        )
    return shape


def validate_waveform_targets(targets: Any, num_sources: int) -> tuple[int, ...]:
    """Validate waveform targets.

    Accepted shapes:
    - [num_sources, num_samples]
    - [batch, num_sources, num_samples]
    """

    shape = _shape_of(targets)
    if not (_is_unbatched_shape(shape, num_sources) or _is_batch_shape(shape, num_sources)):
        raise ValueError(
            "Waveform targets must have shape [num_sources, num_samples] or "
            "[batch, num_sources, num_samples]"
        )
    return shape


def validate_waveform_source(source: Any) -> tuple[int, ...]:
    """Validate one separated waveform source.

    Accepted shapes:
    - [num_samples]
    - [batch, num_samples]
    """

    shape = _shape_of(source)
    valid = len(shape) == 1 or len(shape) == 2
    if not valid:
        raise ValueError("Waveform source must have shape [num_samples] or [batch, num_samples]")
    return shape


def validate_stft_mixture(mixture: Any, num_mics: int) -> tuple[int, ...]:
    """Validate an STFT mixture tensor.

    Accepted shapes:
    - [num_mics, freq_bins, frames, 2]
    - [batch, num_mics, freq_bins, frames, 2]

    The trailing dimension of size 2 stores real and imaginary parts.
    """

    shape = _shape_of(mixture)
    valid = (
        len(shape) == 4 and shape[0] == num_mics and shape[-1] == 2,
        len(shape) == 5 and shape[1] == num_mics and shape[-1] == 2,
    )
    if not any(valid):
        raise ValueError(
            "STFT mixture must have shape [num_mics, freq_bins, frames, 2] or "
            "[batch, num_mics, freq_bins, frames, 2]"
        )
    return shape


def validate_stft_targets(targets: Any, num_sources: int) -> tuple[int, ...]:
    """Validate STFT targets.

    Accepted shapes:
    - [num_sources, freq_bins, frames, 2]
    - [batch, num_sources, freq_bins, frames, 2]
    """

    shape = _shape_of(targets)
    valid = (
        len(shape) == 4 and shape[0] == num_sources and shape[-1] == 2,
        len(shape) == 5 and shape[1] == num_sources and shape[-1] == 2,
    )
    if not any(valid):
        raise ValueError(
            "STFT targets must have shape [num_sources, freq_bins, frames, 2] or "
            "[batch, num_sources, freq_bins, frames, 2]"
        )
    return shape


def validate_stft_source(source: Any) -> tuple[int, ...]:
    """Validate one separated STFT source.

    Accepted shapes:
    - [freq_bins, frames, 2]
    - [batch, freq_bins, frames, 2]
    """

    shape = _shape_of(source)
    valid = (
        len(shape) == 3 and shape[-1] == 2,
        len(shape) == 4 and shape[-1] == 2,
    )
    if not any(valid):
        raise ValueError(
            "STFT source must have shape [freq_bins, frames, 2] or [batch, freq_bins, frames, 2]"
        )
    return shape


def describe_contract(config: ArrayDPSIOConfig) -> dict[str, Any]:
    """Return a machine-readable description of the input/output setup."""

    config.validate()
    if config.signal_format == SignalFormat.WAVEFORM:
        input_shape = "[batch, num_mics, num_samples]"
        target_shape = "[batch, num_sources, num_samples]"
    else:
        input_shape = "[batch, num_mics, freq_bins, frames, 2]"
        target_shape = "[batch, num_sources, freq_bins, frames, 2]"

    return {
        "config": asdict(config),
        "input_shape": input_shape,
        "target_shape": target_shape,
        "defaults": {
            "signal_format": config.signal_format.value,
            "num_mics": config.num_mics,
            "num_sources": config.num_sources,
        },
    }

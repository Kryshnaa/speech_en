from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SignalFormat(str, Enum):
    """Supported signal representations for ArrayDPS."""

    WAVEFORM = "waveform"
    STFT = "stft"


@dataclass(frozen=True, slots=True)
class ArrayDPSIOConfig:
    """I/O contract for an ArrayDPS separation experiment.

    Defaults:
    - STFT input/output representation
    - 6 microphones in the input array
    - 2 output speakers
    - 16 kHz sample rate
    """

    signal_format: SignalFormat = SignalFormat.STFT
    num_mics: int = 6
    num_sources: int = 2
    sample_rate: int = 16_000
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    eps: float = 1e-8

    def validate(self) -> None:
        if self.num_mics < 1:
            raise ValueError("num_mics must be at least 1")
        if self.num_sources < 2:
            raise ValueError("num_sources must be at least 2")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.win_length <= 0:
            raise ValueError("win_length must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    @property
    def summary(self) -> str:
        return (
            f"format={self.signal_format.value}, mics={self.num_mics}, "
            f"sources={self.num_sources}, sr={self.sample_rate}, "
            f"n_fft={self.n_fft}, hop={self.hop_length}, win={self.win_length}"
        )

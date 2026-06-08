"""Small SEGAN-like baseline package.

This package contains a minimal waveform GAN (encoder-decoder generator and
conv1d discriminator) and a dataset helper for manifest-based waveform data.
"""

from .models import Generator, Discriminator
from .dataset import ManifestWaveDataset

__all__ = ["Generator", "Discriminator", "ManifestWaveDataset"]

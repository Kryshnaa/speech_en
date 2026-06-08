from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=31, stride=2, padding=15, use_bn=True):
        super().__init__()
        layers = [nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=31, stride=2, padding=15, use_bn=True):
        super().__init__()
        layers = [nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, output_padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    """Simple encoder-decoder generator with skip connections for waveforms.

    Input: (B, 1, T) waveform tensor. Output: (B, 1, T) enhanced waveform.
    """

    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(1, base_channels, use_bn=False)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)

        self.dec3 = DeconvBlock(base_channels * 8, base_channels * 4)
        self.dec2 = DeconvBlock(base_channels * 4 * 2, base_channels * 2)
        self.dec1 = DeconvBlock(base_channels * 2 * 2, base_channels)

        self.out = nn.Conv1d(base_channels * 2, 1, kernel_size=1)

    def forward(self, x):
        # x: (B,1,T)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self.dec3(e4)
        # align temporal sizes before concatenation (can differ by 1 sample)
        if d3.size(-1) != e3.size(-1):
            m = min(d3.size(-1), e3.size(-1))
            d3 = d3[..., :m]
            e3 = e3[..., :m]
        d3 = torch.cat([d3, e3], dim=1)
        d2 = self.dec2(d3)
        if d2.size(-1) != e2.size(-1):
            m = min(d2.size(-1), e2.size(-1))
            d2 = d2[..., :m]
            e2 = e2[..., :m]
        d2 = torch.cat([d2, e2], dim=1)
        d1 = self.dec1(d2)
        if d1.size(-1) != e1.size(-1):
            m = min(d1.size(-1), e1.size(-1))
            d1 = d1[..., :m]
            e1 = e1[..., :m]
        d1 = torch.cat([d1, e1], dim=1)

        out = self.out(d1)
        return out


class Discriminator(nn.Module):
    """Conv1D patch discriminator producing a single realism score per patch."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=31, stride=4, padding=15),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=31, stride=4, padding=15),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=31, stride=4, padding=15),
            nn.BatchNorm1d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels * 4, 1, kernel_size=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        # x: (B,1,T)
        return self.net(x)

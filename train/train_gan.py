"""Train a minimal SEGAN-style GAN on waveform pairs.

Usage:
  source .venv/bin/activate
  python train/train_gan.py --manifest data/voicebank/manifest_train.json --epochs 5

This is intentionally small and easy to explain line-by-line to a mentor.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from segan import Generator, Discriminator, ManifestWaveDataset


def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def collate_pad(batch):
    # Simple collate that pads to the max length in batch
    mixes = [b["mixture"] for b in batch]
    cleans = [b["clean"] for b in batch]
    max_len = max(x.shape[-1] for x in mixes)
    mix_t = torch.stack([nn.functional.pad(x, (0, max_len - x.shape[-1])) for x in mixes])
    clean_t = torch.stack([nn.functional.pad(x, (0, max_len - x.shape[-1])) for x in cleans])
    return mix_t, clean_t


def train(manifest: str, epochs: int = 10, batch_size: int = 8, lr: float = 2e-4, device: str = "cpu"):
    ds = ManifestWaveDataset(manifest)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)

    G = Generator().to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    optimD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    optimG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))

    real_label = 1.0
    fake_label = 0.0

    for epoch in range(epochs):
        t0 = time.time()
        for i, (mix, clean) in enumerate(dl):
            mix = mix.to(device)
            clean = clean.to(device)

            # -----------------------
            # Update Discriminator
            # -----------------------
            D.zero_grad()
            # real
            out_real = D(clean)
            loss_real = bce(out_real, torch.full_like(out_real, real_label, device=device))
            # fake
            fake = G(mix)
            # ensure fake and clean have same temporal length for reconstruction loss
            if fake.shape[-1] != clean.shape[-1]:
                min_l = min(fake.shape[-1], clean.shape[-1])
                fake = fake[..., :min_l]
                clean = clean[..., :min_l]
            out_fake = D(fake.detach())
            loss_fake = bce(out_fake, torch.full_like(out_fake, fake_label, device=device))
            lossD = (loss_real + loss_fake) * 0.5
            lossD.backward()
            optimD.step()

            # -----------------------
            # Update Generator
            # -----------------------
            G.zero_grad()
            out_fake_for_g = D(fake)
            adv_loss = bce(out_fake_for_g, torch.full_like(out_fake_for_g, real_label, device=device))
            recon_loss = l1(fake, clean)
            lossG = 1e-3 * adv_loss + 1.0 * recon_loss
            lossG.backward()
            optimG.step()

            if i % 10 == 0:
                print(f"E{epoch+1}/{epochs} it {i} | D={lossD.item():.4f} G={lossG.item():.4f} (adv={adv_loss.item():.4f} rec={recon_loss.item():.4f})")

        dt = time.time() - t0
        print(f"Epoch {epoch+1} done in {dt:.1f}s")
        # save checkpoint
        ckpt = Path("checkpoints")
        ckpt.mkdir(exist_ok=True)
        torch.save({"G": G.state_dict(), "D": D.state_dict()}, ckpt / f"gan_epoch{epoch+1}.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()

    train(args.manifest, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)


if __name__ == "__main__":
    main()

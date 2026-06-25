"""
Train a small CNN to predict tunnel coupling t from a 2D sensor signal.

Usage:
    python train.py --data ../data/dataset.h5 --epochs 50
"""

import argparse

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


class CSDDataset(Dataset):
    def __init__(self, h5_path: str):
        with h5py.File(h5_path, "r") as f:
            self.signals = torch.tensor(np.array(f["signals"]), dtype=torch.float32)
            self.labels = torch.tensor(np.array(f["labels"]), dtype=torch.float32)

        # Normalise signals to zero mean, unit std per sample
        mean = self.signals.mean(dim=(-2, -1), keepdim=True)
        std = self.signals.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        self.signals = (self.signals - mean) / std

        # Add channel dim: (N, 1, H, W)
        self.signals = self.signals.unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


class CouplingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def train(data_path: str, epochs: int, batch_size: int, lr: float, save_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CSDDataset(data_path)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = CouplingCNN().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(y)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += loss_fn(model(x), y).item() * len(y)
        val_loss /= n_val

        print(f"Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/dataset.h5")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save", type=str, default="../data/model.pt")
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch_size, args.lr, args.save)

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

        # Normalise per sample to zero mean, unit std
        mean = self.signals.mean(dim=-1, keepdim=True)
        std = self.signals.std(dim=-1, keepdim=True).clamp(min=1e-6)
        self.signals = (self.signals - mean) / std

        # Add channel dim: (N, 1, L)
        self.signals = self.signals.unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


class CouplingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # After 3x MaxPool1d(2) on grid_size=64: 64/8 = 8 spatial positions
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1) * 0.15


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
        all_preds = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                val_loss += loss_fn(preds, y).item() * len(y)
                all_preds.append(preds.cpu())
        val_loss /= n_val
        preds_cat = torch.cat(all_preds)
        pred_std = preds_cat.std().item()
        pred_mean = preds_cat.mean().item()

        print(f"Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  pred_mean={pred_mean:.4f}  pred_std={pred_std:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/dataset.h5")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save", type=str, default="../data/model.pt")
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch_size, args.lr, args.save)

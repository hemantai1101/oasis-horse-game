"""
train.py — PyTorch training script for the Squirrel Oasis value network.

Model:  Input(21) → Dense(128, ReLU) → BatchNorm → Dense(64, ReLU) → BatchNorm → Dense(1, Tanh)
Loss:   MSE
Optim:  Adam (lr=1e-3, weight_decay=1e-4, with cosine LR schedule)

Usage:
  python train.py                          # train with defaults
  python train.py --epochs 50             # more epochs
  python train.py --data data/games.jsonl # custom data file

Output:
  models/model.pt       — saved PyTorch model (best validation loss)
  models/model_latest.pt — saved PyTorch model (every 5 epochs)
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ===== Model =====

class ValueNet(nn.Module):
    """
    Small MLP: 21 inputs → 128 → 64 → 1 (tanh output in [-1, 1]).
    Output = +1 means current player wins; -1 means current player loses.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(21, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # shape: (batch,)


# ===== Dataset =====

class GameDataset(Dataset):
    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f'{path} not found.\n'
                f'Run: python generate_games.py to create it first.'
            )

        # First pass: count lines to pre-allocate memory
        print(f"Counting records in {path}...")
        with open(path) as f:
            num_lines = sum(1 for line in f if line.strip())

        # Pre-allocate numpy arrays
        features = np.zeros((num_lines, 21), dtype=np.float32)
        labels = np.zeros(num_lines, dtype=np.float32)

        # Second pass: fill the arrays
        print(f"Loading {num_lines:,} records...")
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                features[i] = obj['features']
                labels[i] = obj['label']

        self.X = torch.from_numpy(features)
        self.y = torch.from_numpy(labels)
        print(f'Finished loading data.')


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===== Training Loop =====

def train(args):
    script_dir = Path(__file__).parent

    data_path  = Path(args.data) if args.data else script_dir / 'data' / 'games.jsonl'
    model_dir  = script_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    best_save_path = model_dir / 'model.pt'
    latest_save_path = model_dir / 'model_latest.pt'


    # --- Dataset ---
    dataset  = GameDataset(data_path)
    n_val    = max(1, int(len(dataset) * 0.05))   # 5% validation
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 4, shuffle=False,
                              num_workers=0)

    # --- Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model  = ValueNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params:,}')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = math.inf
    start_time = time.time()

    print(f'\nTraining for {args.epochs} epochs...\n')
    print(f'{"Epoch":>5}  {"Train Loss":>10}  {"Val Loss":>10}  {"LR":>8}')
    print('-' * 42)

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= n_train

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item() * len(y_batch)
        val_loss /= n_val

        lr_now = scheduler.get_last_lr()[0]
        print(f'{epoch:>5}  {train_loss:>10.5f}  {val_loss:>10.5f}  {lr_now:>8.2e}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_save_path)

        if epoch % 5 == 0:
            torch.save(model.state_dict(), latest_save_path)

        scheduler.step()

    elapsed = time.time() - start_time
    print(f'\nBest val loss: {best_val_loss:.5f}')
    print(f'Saved best model to: {best_save_path}')
    print(f'Total time: {elapsed:.1f}s')
    print('\nNext step:  python export_weights.py')


# ===== Entry Point =====

def main():
    parser = argparse.ArgumentParser(description='Train the Squirrel Oasis value network')
    parser.add_argument('--data',         type=str,   default=None,
                        help='Path to games.jsonl (default: training/data/games.jsonl)')
    parser.add_argument('--epochs',       type=int,   default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size',   type=int,   default=1024,
                        help='Batch size (default: 1024)')
    parser.add_argument('--lr',           type=float, default=1e-3,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Adam weight decay (default: 1e-4)')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

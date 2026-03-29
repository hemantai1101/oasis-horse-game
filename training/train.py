"""
train.py — PyTorch training script for the Squirrel Oasis value network.

Model:  Input(110) → Dense(256, ReLU) → BatchNorm → Dense(128, ReLU) → BatchNorm → Dense(1, Tanh)
Loss:   MSE
Optim:  Adam (lr=1e-3, weight_decay=1e-4, with cosine LR schedule)

Usage:
  python train.py                                      # train with defaults
  python train.py --epochs 100                         # custom epoch count
  python train.py --data data/games.jsonl              # custom data file
  python train.py --output models/model_run008.pt      # custom output file
  python train.py --resume models/model_run008_checkpoint.pt --epochs 200  # continue to epoch 200

Output:
  --output sets the full path for the best model file.
  Default: training/models/model.pt
  model_latest.pt and model_checkpoint.pt are always saved in the same directory.

Resuming:
  --resume loads a checkpoint and continues training.
  --epochs must be greater than the checkpoint epoch (it is the new TOTAL target).
  Example: checkpoint at epoch 100, pass --epochs 200 to train 100 more epochs.
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
    MLP: input_size → 256 → 128 → 1 (tanh output in [-1, 1]).
    Output = +1 means current player wins; -1 means current player loses.
    input_size=105 for Run 001-006, 110 for Run 007-012, 115 for Run 013+.
    """
    def __init__(self, input_size=115):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
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

        def _valid(line):
            line = line.strip()
            if not line or '\x00' in line:
                return False
            try:
                json.loads(line)
                return True
            except json.JSONDecodeError:
                return False

        # First pass: count valid lines to pre-allocate memory
        print(f"Counting records in {path}...")
        with open(path) as f:
            num_lines = sum(1 for line in f if _valid(line))

        # Detect feature size from first valid record
        input_size = 115  # default for Run 013+
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and '\x00' not in line:
                    try:
                        first = json.loads(line)
                        input_size = len(first['features'])
                        break
                    except (json.JSONDecodeError, KeyError):
                        pass
        self.input_size = input_size

        # Pre-allocate numpy arrays
        features = np.zeros((num_lines, input_size), dtype=np.float32)
        labels = np.zeros(num_lines, dtype=np.float32)

        # Second pass: fill the arrays
        print(f"Loading {num_lines:,} records... (input_size={input_size})")
        skipped = 0
        with open(path) as f:
            j = 0
            for line in f:
                line = line.strip()
                if not line or '\x00' in line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                features[j] = obj['features']
                labels[j] = obj['label']
                j += 1
        if skipped:
            print(f"Skipped {skipped:,} corrupt lines.")

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

    data_path      = Path(args.data) if args.data else script_dir / 'data' / 'games.jsonl'
    best_save_path       = Path(args.output) if args.output else script_dir / 'models' / 'model.pt'
    model_dir            = best_save_path.parent
    model_dir.mkdir(parents=True, exist_ok=True)
    latest_save_path     = model_dir / (best_save_path.stem + '_latest.pt')
    checkpoint_save_path = model_dir / (best_save_path.stem + '_checkpoint.pt')


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
    model  = ValueNet(input_size=dataset.input_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params:,}')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- CPU thread count ---
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    print(f'CPU threads: {torch.get_num_threads()}')

    # --- Resume from checkpoint if requested ---
    start_epoch   = 0
    best_val_loss = math.inf

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch   = ckpt['epoch']
        best_val_loss = ckpt['best_val_loss']
        if args.epochs <= start_epoch:
            raise ValueError(f'--epochs {args.epochs} must be greater than checkpoint epoch {start_epoch}')
        print(f'Resumed from epoch {start_epoch} — best val loss so far: {best_val_loss:.5f}')
        print(f'Continuing to epoch {args.epochs} ({args.epochs - start_epoch} more epochs)\n')

    # Cosine schedule over total epochs; last_epoch positions LR correctly on resume
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, last_epoch=start_epoch - 1
    )

    start_time = time.time()

    print(f'\nTraining for {args.epochs} epochs...\n')
    print(f'{"Epoch":>5}  {"Train Loss":>10}  {"Val Loss":>10}  {"LR":>8}')
    print('-' * 42)

    for epoch in range(start_epoch + 1, args.epochs + 1):
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

        scheduler.step()

        # Save full checkpoint every 5 epochs for resume support
        if epoch % 5 == 0:
            torch.save(model.state_dict(), latest_save_path)
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss':        best_val_loss,
            }, checkpoint_save_path)

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
    parser.add_argument('--output',       type=str,   default=None,
                        help='Full path for best model file (default: training/models/model.pt)')
    parser.add_argument('--resume',       type=str,   default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--num-threads',  type=int,   default=None,
                        help='PyTorch CPU thread count (default: PyTorch auto). Try 8-16 on many-core machines.')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

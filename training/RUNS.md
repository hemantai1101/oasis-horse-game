# Training Runs

A log of every data generation + training run for reproducibility and comparison.

**Key metric to compare:** Best val loss (lower = better model)

---

## Run 001 — 2026-03-13

### Data Generation
```bash
pypy3 training/generate_games.py \
  --n-games 10000 \
  --workers 7 \
  --depth 3 \
  --time-limit 0.2 \
  --epsilon 0.15
```

| Parameter | Value | Note |
|-----------|-------|------|
| n-games | 10,000 | |
| workers | 7 | VM CPU count - 1 |
| depth | 3 | Faster than default 5, slightly weaker teacher |
| time-limit | 0.2s | Per move |
| epsilon | 0.15 | 15% chance of picking from top-3 moves (adds variety) |
| augmentation | 2× | 180° rotation applied automatically |

Records written: 975,306 records

### Training
```bash
python3 training/train.py \
  --epochs 50 \
  --batch-size 2048 \
  --lr 1e-3 \
  --weight-decay 1e-4
```

| Parameter | Value | Note |
|-----------|-------|------|
| epochs | 50 | More than default 30 to compensate for depth-3 data |
| batch-size | 2048 | Larger than default 1024, faster on VM |
| lr | 1e-3 | Default |
| weight-decay | 1e-4 | Default |

Best val loss: **0.60162** (epoch 48)
Train loss at epoch 50: 0.58047
Converged at epoch: ~48 — val loss flatlined as LR approached zero (cosine schedule exhausted)
Time: 172.4s

Notes:
- No overfitting — train/val gap stayed small and stable throughout (0.022 at epoch 50)
- Val loss noisy in epochs 7–20 (normal at high LR), stabilised cleanly after epoch 30
- Still slowly decreasing at epoch 48 — model had not fully plateaued, just ran out of LR budget
- Baseline is good; **main bottleneck is data volume**, not model capacity or hyperparams
- Next run: increase to 50k games to see if val loss drops below 0.58

### Model
Saved to: `training/models/model.pt`
Exported to: `public/model/weights.json`

---

## Run 002 — *(pending)*

**Goal:** More data to push val loss below 0.58.

### Data Generation
```bash
pypy3 training/generate_games.py \
  --n-games 50000 \
  --workers 14 \
  --depth 3 \
  --time-limit 0.2 \
  --epsilon 0.15 \
  --output training/data/games_50k_3_.2_.15.jsonl
```

| Parameter | Value | Note |
|-----------|-------|------|
| n-games | 50,000 | 5× Run 001 |
| workers | 14 | c2-standard-16 |
| depth | 3 | Same as Run 001 |
| time-limit | 0.2s | Same as Run 001 |
| epsilon | 0.15 | Same as Run 001 |
| augmentation | 2× | 180° rotation applied automatically |

Records written: *(pending)*

### Training
```bash
python3 training/train.py \
  --epochs 50 \
  --batch-size 2048 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --data training/data/games_50k_3_.2_.15.jsonl
```

Best val loss: *(pending)*
Converged at epoch: *(pending)*
Notes: *(pending)*

### Model
Saved to: `training/models/model.pt`
Exported to: `public/model/weights.json`

---

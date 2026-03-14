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

## Run 002 — 2026-03-14

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

Games completed: 49,653 / 50,000 (347 skipped, 0.7%)
Records written: 4,838,608
Generation time: 34,286.8s (~9.5 hrs)

### Training
```bash
python3 training/train.py \
  --epochs 50 \
  --batch-size 2048 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --data training/data/games_50k_3_.2_.15.jsonl
```

| Parameter | Value | Note |
|-----------|-------|------|
| epochs | 50 | Same as Run 001 |
| batch-size | 2048 | Same as Run 001 |
| lr | 1e-3 | Default |
| weight-decay | 1e-4 | Default |

Best val loss: **0.61344** (epoch 50)
Train loss at epoch 50: 0.61031
Train/val gap at epoch 50: 0.00313 (no overfitting)
Converged at epoch: **not converged — still declining at epoch 50**
Training time: 13,756.1s (~3.8 hrs)

Val loss curve (selected epochs):
| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 0.71751 | 0.68923 |
| 10 | 0.64238 | 0.65005 |
| 25 | 0.62559 | 0.62758 |
| 35 | 0.61728 | 0.61897 |
| 42 | 0.61267 | 0.61480 |
| 50 | 0.61031 | 0.61344 |

Notes:
- Val loss was **still declining** at epoch 50 (same LR-exhaustion pattern as Run 001)
- Cosine LR schedule ran out of budget before convergence (LR at epoch 50: 9.87e-07 ≈ 0)
- No overfitting — train/val gap only 0.003 throughout, safe to train longer
- Raw val loss (0.613) appears higher than Run 001 (0.602) but is **not directly comparable** — each run's val set is a random split of its own dataset; 50k data is more diverse so the validation problem is harder
- Gradient step count: ~118,100 total (vs ~23,800 in Run 001) — 5× more updates, model still learning
- Main bottleneck: **epoch count / LR budget**, not data volume or model capacity
- Next run: increase to `--epochs 100` on same 50k dataset — model should keep improving

### Model
Saved to: `training/models/model.pt`
Exported to: `public/model/weights.json`

---

## Run 003 — 2026-03-14

**Goal:** Let the model converge — same 50k data, double the epochs (100).

### Training
```bash
python3 training/train.py \
  --epochs 100 \
  --batch-size 2048 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --data training/data/games_50k_3_.2_.15.jsonl
```

| Parameter | Value | Note |
|-----------|-------|------|
| epochs | 100 | 2× Run 002 — give cosine schedule more budget |
| batch-size | 2048 | Same |
| lr | 1e-3 | Same |
| weight-decay | 1e-4 | Same |
| data | games_50k_3_.2_.15.jsonl | Same as Run 002 |

Best val loss: **0.60809** (epoch 100)
Train loss at epoch 100: 0.60495
Train/val gap: 0.00314 (no overfitting)
Converged at epoch: **~95** — improvement from epoch 95→100 only 0.00028, essentially flat
Training time: 1,920.7s (~32 min)

Val loss curve (selected epochs):
| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 0.71771 | 0.68189 |
| 25 | 0.63098 | 0.63785 |
| 50 | 0.61967 | 0.62451 |
| 75 | 0.60985 | 0.61198 |
| 90 | 0.60591 | 0.60878 |
| 95 | 0.60519 | 0.60837 |
| 100 | 0.60495 | 0.60809 |

Notes:
- Model properly converged this time — val loss essentially flat from epoch ~95 onward
- No overfitting — train/val gap stable at 0.003 throughout
- More epochs would give diminishing returns (< 0.001 gain expected)
- Val loss 0.60809 vs Run 001's 0.60162 — not directly comparable (different val sets, 50k data is more diverse)
- **This is the ceiling for depth-3 data + current architecture** — further gains require stronger teacher data
- Next lever: generate 50k games at depth=4 to give the model a smarter teacher

### Model
Saved to: `training/models/model.pt` → backed up as `training/models/model_run003.pt`
Exported to: `public/model/weights.json`

---

## Run 004 — *(pending)*

**Goal:** Stronger teacher — regenerate 50k games at depth=4, retrain.

### Data Generation
```bash
pypy3 training/generate_games.py \
  --n-games 50000 \
  --workers 14 \
  --depth 4 \
  --time-limit 0.5 \
  --epsilon 0.15 \
  --output training/data/games_50k_4_.5_.15.jsonl
```

| Parameter | Value | Note |
|-----------|-------|------|
| n-games | 50,000 | Same as Run 002/003 |
| workers | 14 | c2-standard-16 |
| depth | 4 | ↑ from 3 — stronger teacher |
| time-limit | 0.5s | ↑ from 0.2s — depth-4 needs more time per move |
| epsilon | 0.15 | Same |

### Training
```bash
python3 training/train.py \
  --epochs 100 \
  --batch-size 2048 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --data training/data/games_50k_4_.5_.15.jsonl
```

Best val loss: *(pending)*
Converged at epoch: *(pending)*
Notes: *(pending)*

### Model
Saved to: `training/models/model.pt`
Exported to: `public/model/weights.json`

---

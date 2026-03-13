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

Records written: 1005544

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

Best val loss: *(pending)*
Converged at epoch: *(pending)*
Notes: *(pending — did loss plateau or still decreasing at epoch 50?)*

### Model
Saved to: `training/models/model.pt`
Exported to: `public/model/weights.json`

---

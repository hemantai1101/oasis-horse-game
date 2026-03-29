# Squirrel Oasis — Training Pipeline

This folder contains everything needed to train the neural network that powers the browser AI.
Trained weights (`public/model/weights.json`) replace the handcrafted evaluation heuristics
inside the minimax search — the search structure itself stays intact.

---

## How It Works

1. **Generate games** — A Python minimax AI plays against itself, recording every board position
   and the final outcome (+1 win / −1 loss).
2. **Train** — A value network learns to predict outcome from board features (110 inputs).
3. **Export** — Weights are written to `public/model/weights.json` and loaded by the browser AI.

The NN acts as the leaf evaluator inside the browser's alpha-beta minimax. Because NN inference
is ~1000× faster than full minimax search, the same time budget allows much deeper search.

---

## Prerequisites

Install Python dependencies (CPU-only, no GPU needed):

```bash
pip install torch numpy
```

For game generation, use **PyPy3** on the training machine — it's 5–10× faster for this
CPU-bound workload. The training step (`train.py`) requires CPython with PyTorch.

---

## File Overview

| File | Purpose |
|------|---------|
| `game.py` | Python port of `public/game.js` — board, moves, zones |
| `minimax.py` | Python port of `public/ai-worker.js` — minimax search with alpha-beta |
| `generate_games.py` | Self-play game generator; writes `.jsonl` records |
| `train.py` | PyTorch training loop; saves best checkpoint by val loss |
| `export_weights.py` | Converts `model.pt` → `../public/model/weights.json` |
| `requirements.txt` | Python dependencies |
| `data/` | Generated JSONL files (gitignored) |
| `models/` | Saved `.pt` checkpoints (gitignored) |
| `runs/` | Per-run documentation (`run_NNN.md`) |
| `RUNS.md` | Master run log with val loss progression |

---

## Script Reference

### generate_games.py

Plays self-play games between minimax AIs at configurable depth and writes positions +
outcomes to a `.jsonl` file. Each game is also augmented with a 180° board rotation
(doubles the dataset size for free).

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--n-games` | 10000 | Total games to attempt |
| `--output` | `data/games.jsonl` | Output file path |
| `--depth` | 5 | Teacher AI search depth. Higher = stronger data, slower generation |
| `--time-limit` | 1.0 | Seconds per move. Use `3.0` for depth=5 on the training machine |
| `--epsilon` | 0.15 | Exploration rate. `0.15` adds position diversity without noise |
| `--workers` | cpu_count − 1 | Parallel workers. Match your core count |
| `--seed` | 42 | Fix for reproducibility |

**Run 011 command (current):**

```bash
pypy3 training/generate_games.py \
  --n-games 50000 \
  --depth 5 \
  --time-limit 3 \
  --epsilon 0.15 \
  --output training/data/games_50k_5_3_.15_v4.jsonl
```

**Smoke test:**

```bash
pypy3 training/generate_games.py --n-games 20 --depth 3
```

**Skip rate note:** At depth=5 with `MAX_MOVES=600`, expect ~14% of games to be skipped
(games that don't resolve within the move cap). This is normal — surviving games are
high-quality decisive games.

---

### train.py

Trains the value network on a `.jsonl` file. Saves the best checkpoint (by val loss) to
the path given by `--output`, and a rolling checkpoint every 5 epochs as `*_checkpoint.pt`.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--data` | `training/data/games.jsonl` | Input JSONL file |
| `--output` | `training/models/model.pt` | Full path for best model checkpoint |
| `--epochs` | 30 | Training epochs. Use 150 for large depth=5 datasets |
| `--batch-size` | 1024 | Increase to 2048 for datasets > 1M records |
| `--lr` | 0.001 | Learning rate |
| `--resume` | — | Path to `*_checkpoint.pt` to resume interrupted training |

**Run 011 training command:**

```bash
~/venv/bin/python training/train.py \
  --data training/data/games_50k_5_3_.15_v4.jsonl \
  --output training/models/model_run011.pt \
  --epochs 150
```

**Reading the output:**
- Both losses decreasing → keep training
- Val loss rising while train loss falls → overfitting; stop or add `--weight-decay`
- Both flat from epoch 1 → dataset too small or LR too low

---

### export_weights.py

Converts a `.pt` checkpoint to the JSON format loaded by the browser.

```bash
python training/export_weights.py \
  --model training/models/model_run011.pt \
  --output public/model/weights.json
```

Then deploy:

```bash
firebase deploy --only hosting
```

---

## State Representation

Each board position is encoded as **110 floating-point numbers** (feature version 4):

```
Per horse × 20 horses = indices 0–99  (P1 first, then P2, sorted by col/row):
  base+0  col_norm        = (col - 1) / 10              ∈ [0, 1]
  base+1  row_norm        = (row - 1) / 10              ∈ [0, 1]
  base+2  dist_to_center  = (|col-6| + |row-6|) / 10   ∈ [0, 1]
  base+3  on_axis         = 1.0 if col=6 or row=6       ∈ {0, 1}
  base+4  path_clear      = 1.0 if on_axis AND no blocker between piece and center

Global features [100–109]:
  [100]  backstop (6,5)       +1.0=current player, -1.0=opponent, 0.0=empty
  [101]  backstop (6,7)       same
  [102]  backstop (5,6)       same
  [103]  backstop (7,6)       same
  [104]  player indicator     0.0 = P1's turn, 1.0 = P2's turn
  [105]  my_winning_threats  / 2   — pieces with on_axis + clear path + backstop set
  [106]  opp_winning_threats / 2   — same for opponent
  [107]  my_horses_at_home   / 10  — pieces still in starting corner regions
  [108]  opp_horses_at_home  / 10  — opponent pieces still in starting corners
  [109]  my_pieces_blocking_opp / 5 — axis pieces blocking an opponent piece further back
```

---

## Model Architecture

```
Input (110) → Linear(256) → ReLU → BatchNorm
            → Linear(128) → ReLU → BatchNorm
            → Linear(1)   → Tanh
```

Output ∈ [−1, +1]. Positive = good for the player to move.
Trained with Adam + cosine LR annealing + L2 weight decay.
Total parameters: ~46,000.

---

## Training Target

For every position in a recorded game:
- **+1.0** if the player to move eventually won
- **−1.0** if the player to move eventually lost

MSE loss. The network learns a value function used as the leaf evaluator in minimax.

---

## GCP Training Machine

**Project:** `oasis-horse-game` | **Machine:** `power-trainer` | **Zone:** `us-central1-f` | **User:** `trainer`

---

### SSH into the machine

```bash
gcloud compute ssh trainer@power-trainer \
  --project=oasis-horse-game \
  --zone=us-central1-f \
  --tunnel-through-iap
```

---

### Push training code (rsync)

One-time SSH config setup (re-run if the machine restarts and IP changes):

```bash
gcloud compute config-ssh --project=oasis-horse-game
```

Then sync only changed files:

```bash
rsync -avz --progress \
  --exclude='data/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='models/' \
  ./training \
  trainer@power-trainer.us-central1-f.oasis-horse-game:~/oasis-horse-game/
```

---

### Run generation inside screen

```bash
screen -S gen

pypy3 training/generate_games.py \
  --n-games 50000 \
  --depth 5 \
  --time-limit 3 \
  --epsilon 0.15 \
  --output training/data/games_50k_5_3_.15_v4.jsonl

# Detach: Ctrl+A then D
# Reattach: screen -r gen
```

---

### Monitor progress

```bash
# Check line count (records written so far)
wc -l ~/oasis-horse-game/training/data/games_50k_5_3_.15_v4.jsonl

# Check if still running
lsof ~/oasis-horse-game/training/data/games_50k_5_3_.15_v4.jsonl
```

---

### Download the data file

```bash
rsync -avz --progress \
  trainer@power-trainer.us-central1-f.oasis-horse-game:~/oasis-horse-game/training/data/games_50k_5_3_.15_v4.jsonl \
  ./training/data/
```

---

### First-time machine setup

```bash
sudo add-apt-repository universe
sudo apt-get update && sudo apt-get install -y pypy3 screen
python3 -m venv ~/venv && ~/venv/bin/pip install torch numpy
```

---

## Troubleshooting

**`FileNotFoundError: data/games.jsonl not found`**
→ Run `generate_games.py` first.

**`FileNotFoundError: models/model.pt not found`**
→ Run `train.py` first.

**`[AI] No weights.json found — using heuristic evaluation.`** in browser console
→ Export weights and serve via HTTP. File:// protocol blocks `fetch()`.
  Local test: `python3 -m http.server 8080 --directory public` then open `localhost:8080`.

**Training loss not decreasing**
→ Dataset likely too small. Try more games or more epochs. Check for null-byte
  corruption if data was concatenated from multiple files: `grep -c $'\x00' data/file.jsonl`

**JSONDecodeError on training**
→ Blank lines or null bytes in the JSONL. The loader skips these automatically,
  but a high count suggests file corruption.

**20%+ skip rate during generation**
→ `MAX_MOVES` cap too low for the depth. Currently set to 600 for depth=5.
  Increase if skip rate is above 20%.

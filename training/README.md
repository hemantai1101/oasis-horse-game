# Squirrel Oasis — Training Pipeline

This folder contains everything needed to train a neural network that improves the
browser AI. The trained weights (`public/model/weights.json`) replace the handcrafted
evaluation heuristics in `public/ai.js` — the minimax search structure stays intact.

---

## Prerequisites

**Python 3.9 or newer** is required. Install the two dependencies:

```bash
pip install torch numpy
```

No GPU is needed. The model is tiny (~11,500 parameters) and trains in minutes on any laptop.

---

## Quickstart

Run these three commands from the **repository root** (or adjust paths accordingly):

```bash
# 1. Generate self-play games  (~10–30 min for 10,000 games)
python training/generate_games.py

# 2. Train the neural network  (~1–5 min)
python training/train.py

# 3. Export weights to the browser  (seconds)
python training/export_weights.py
```

Then commit and push `public/model/weights.json`:

```bash
git add public/model/weights.json
git commit -m "Add trained NN weights (stage 1)"
git push
```

Open the game in a browser. The AI will automatically load the neural net weights.
Check the browser DevTools console for: `[AI] Neural net weights loaded.`

---

## Smoke Test (fast, before running the full pipeline)

```bash
python training/generate_games.py --n-games 100
# Should produce ~5,000 lines in training/data/games.jsonl

python training/train.py
# Should show decreasing loss over 30 epochs

python training/export_weights.py
# Should produce public/model/weights.json (~100 KB)
```

---

## File Overview

| File | Purpose |
|------|---------|
| `game.py` | Python port of `public/game.js` — board, moves, zones |
| `minimax.py` | Python port of `public/ai.js` — backstop evaluation, minimax search |
| `generate_games.py` | Self-play runner; writes `data/games.jsonl` |
| `train.py` | PyTorch training loop; writes `models/model.pt` (best val loss) and `models/model_latest.pt` (every 5 epochs) |
| `export_weights.py` | Converts `model.pt` → `../public/model/weights.json` |
| `requirements.txt` | Python dependencies (torch, numpy) |
| `data/` | Generated game records (gitignored — create by running generate_games.py) |
| `models/` | Saved model checkpoints (gitignored) |

---

## Script Reference

### generate_games.py

Plays self-play games between minimax AIs and writes board positions + outcomes to a `.jsonl` file. Each game also produces a 180° rotated copy (augmentation), so the output is always 2× the number of positions played.

| Parameter | Default | When to change |
|-----------|---------|----------------|
| `--n-games` | 10000 | Increase for a larger dataset (better generalisation). Use 10–100 for smoke tests. |
| `--output` | `data/games.jsonl` | Set a descriptive name per run so files don't get overwritten, e.g. `data/games_10k_d3.jsonl`. |
| `--seed` | 42 | Change to get a different dataset with the same settings. Keep fixed for reproducibility. |
| `--workers` | cpu_count − 1 | Match your core count. On c2-standard-16 (VM) use `14`. On a 4-core laptop use `3`. |
| `--depth` | 5 | Controls AI strength and generation speed. `3` = fast & good enough. `5` = slower but higher quality data. |
| `--time-limit` | 1.0 | Max seconds per move. Lower = faster generation. `0.2` is a good balance on the VM. |
| `--epsilon` | 0.1 | Exploration rate (0.0–1.0). `0.0` = pure minimax (repetitive games). `0.15` = occasional random moves for position diversity. |

**Example commands:**

```bash
# Quick smoke test — 10 games, verify everything works
pypy3 training/generate_games.py --n-games 10 --workers 3 --depth 3

# Standard production run — VM with 16 cores
pypy3 training/generate_games.py \
  --n-games 10000 --workers 14 --depth 3 --time-limit 0.2 --epsilon 0.15

# Large dataset for stage 2+ — more games, deeper search
pypy3 training/generate_games.py \
  --n-games 50000 --workers 14 --depth 3 --time-limit 0.2 --epsilon 0.15 \
  --output data/games_50k.jsonl

# Reproducible run — fix the seed
pypy3 training/generate_games.py --n-games 10000 --seed 123
```

> **Note:** Use `pypy3` on the VM (faster CPU-bound execution). Use `python3` locally if PyPy is not installed.

---

### train.py

Trains the value network on a `.jsonl` file. Saves the best checkpoint (by validation loss) to `models/model.pt` and a rolling checkpoint every 5 epochs to `models/model_latest.pt`.

| Parameter | Default | When to change |
|-----------|---------|----------------|
| `--data` | `training/data/games.jsonl` | Point to a specific file if you have multiple datasets. |
| `--epochs` | 30 | Increase to `50` if val loss is still decreasing at epoch 30. Stop earlier if val loss plateaus or rises (overfitting). |
| `--batch-size` | 1024 | Increase to `2048` for datasets > 500k records (faster per epoch). Lower if you run out of memory. |
| `--lr` | 0.001 | Lower to `5e-4` or `1e-4` if training loss is unstable or spiky. |
| `--weight-decay` | 1e-4 | Increase if the model overfits (val loss rises while train loss falls). |

**Example commands:**

```bash
# Standard run — defaults are good for most cases
python3 training/train.py

# Longer training on a large dataset
python3 training/train.py --epochs 50 --batch-size 2048 \
  --data training/data/games_50k.jsonl

# Tune if training is unstable
python3 training/train.py --epochs 50 --lr 5e-4 --weight-decay 1e-3
```

**Reading the output:** Each epoch prints `train_loss` and `val_loss`.
- Both decreasing → good, keep training
- `val_loss` rising while `train_loss` falls → overfitting; stop or increase `--weight-decay`
- Both flat from epoch 1 → learning rate too low, or dataset too small

---

## State Representation

Each board position is encoded as **110 floating-point numbers**:

```
Per horse × 20 horses = indices 0–99  (P1 first, then P2, each sorted by col/row):
  base+0  col_norm        = (col - 1) / 10             ∈ [0, 1]
  base+1  row_norm        = (row - 1) / 10             ∈ [0, 1]
  base+2  dist_to_center  = (|col-6| + |row-6|) / 10  ∈ [0, 1]
  base+3  on_axis         = 1.0 if col=6 or row=6      ∈ {0, 1}
  base+4  path_clear      = 1.0 if on_axis AND path to center fully clear, else 0.0

Global features [100–109]:
  [100]  backstop (6,5)  +1.0=current player, -1.0=opponent, 0.0=empty
  [101]  backstop (6,7)  same
  [102]  backstop (5,6)  same
  [103]  backstop (7,6)  same
  [104]  player indicator: 0.0 = P1's turn, 1.0 = P2's turn
  [105]  my_winning_threats  / 2   — horses with on_axis + clear path + my backstop set
  [106]  opp_winning_threats / 2   — same for opponent
  [107]  my_horses_at_home   / 10  — my pieces still in starting corner regions
  [108]  opp_horses_at_home  / 10  — opponent pieces still in starting corners
  [109]  my_pieces_blocking_opp / 5 — my axis pieces blocking an opp piece further back
```

Each player has **10 horses** across 2 corners (e.g. P1 = TL + BR corners, 5 horses each).
Pieces are sorted so the same board state always produces the same feature vector.
The backstop cells are the 4 cells adjacent to center — a horse needs to occupy one of these
to stop an attacker from sliding through center (the win condition).

---

## Training Target

For every position in a recorded game:
- **Label +1.0** if the player whose turn it was eventually **won**
- **Label −1.0** if they eventually **lost**

This is a *value function* — it estimates "how good is this position for the player to move?"
It replaces the handcrafted `evaluate()` in the minimax search as the leaf evaluation.

---

## Model Architecture

```
Input (110) → Linear(256) + ReLU + BatchNorm → Linear(128) + ReLU + BatchNorm → Linear(1) + Tanh
```

Output is in **[−1, +1]**. Positive = good for the player to move; negative = bad.

Training uses Adam with cosine LR annealing and L2 weight decay (`weight_decay=1e-4`).

---

## Data Generation Mix

Games are played between minimax AIs of varying strength to create diverse data:

| Proportion | P1 | P2 | Purpose |
|-----------|----|----|---------|
| 60% | depth-3 | depth-3 | High-quality expert games |
| 15% | depth-2 | depth-3 | P1 makes occasional mistakes |
| 15% | depth-3 | depth-2 | P2 makes occasional mistakes |
| 5%  | random  | depth-3 | Diverse opening positions |
| 5%  | depth-3 | random  | Diverse opening positions |

Each game is also augmented with a 180° board rotation (preserves player-corner
assignments: TL↔BR and TR↔BL stay with the same owner), doubling the dataset size.

---

## Iteration Roadmap

The training pipeline is designed to improve in stages. Each stage produces a
stronger AI than the previous one.

### Stage 1 — Supervised Learning from Minimax (this stage)

**Data source:** Python minimax AI (depth 5 by default) plays against itself.
**Constraint:** The neural net is bounded by the minimax teacher's quality.
**Practical gain:** NN inference is ~1000× faster than minimax search, so the same
                    NN can be used inside a *deeper* minimax search tree.

```bash
python training/generate_games.py --n-games 10000
python training/train.py
python training/export_weights.py
```

**Expected strength:** Roughly equivalent to depth-3 minimax, but evaluations are instant.

---

### Stage 2 — Deeper Minimax with NN Evaluation

**No retraining needed.** Increase the search depth in `public/ai.js` from 3 to 5 or 6:

```js
// In getAIMove(), change:
const val = minimax(child, 3, ...   // ← change to 5 or 6
```

The NN evaluates leaf nodes in < 1 ms instead of the heuristic's microseconds, so
the time budget (750 ms) is still respected at greater depths.

**Expected strength:** Noticeably stronger than stage 1 — the AI "sees further ahead."

---

### Stage 3 — Self-Play Training (break the minimax ceiling)

**This is where the AI can surpass the minimax teacher.**

Instead of using the Python minimax to generate games, use the *trained neural net*
guided by Monte Carlo Tree Search (MCTS). The net plays against itself, discovers
strategies the minimax never found, and trains on those games.

Concretely: update `generate_games.py` to import the trained model and replace
`get_best_move()` calls with MCTS rollouts guided by the NN value function.

```bash
# After stage 1 is complete:
python training/generate_games_mcts.py --n-games 20000  # (stage 3 script, not yet written)
python training/train.py --data data/games_mcts.jsonl
python training/export_weights.py
```

Re-run with the new model as the data generator. Iterate until strength plateaus.

**Expected strength:** Potentially much stronger than any fixed-depth minimax search.

---

### Stage 4 — AlphaZero-Lite (continuous self-improvement)

Automate the stage-3 loop:

1. Generate N games using current model + MCTS
2. Train new model on those games
3. Evaluate new model vs old model (acceptance test: win rate > 55%)
4. Replace old model with new model if accepted
5. Go to step 1

Each iteration, the data generator (the model itself) improves, creating a
positive feedback loop. This is the same principle as AlphaGo Zero and AlphaZero.

**Expected strength:** Converges to near-optimal play for the game tree complexity.

---

## Automated GCP Workflow (fire-and-forget)

Two scripts in `scripts/` let you kick off a generation run from your laptop
and walk away. The VM starts, generates games, uploads to GCS, then shuts
itself down automatically to avoid idle billing.

### One-time setup

**1. Create a GCS bucket** (run once from your local machine):
```bash
gsutil mb -l us-central1 gs://YOUR_BUCKET_NAME
```

**2. Grant the VM write access to the bucket:**
```bash
# Get the VM's service account email
gcloud compute instances describe oasis-budget-worker \
  --project=ff-ml-project --zone=us-central1-c \
  --format='value(serviceAccounts[0].email)'

# Grant Storage Object Admin on the bucket
gsutil iam ch serviceAccount:SA_EMAIL:objectAdmin gs://YOUR_BUCKET_NAME
```

**3. Set your bucket name** in `scripts/generate_and_upload.sh`:
```bash
BUCKET="gs://YOUR_BUCKET_NAME/training-data"   # ← update this line
```

---

### Running a generation job

```bash
# From your local machine — fire and forget
bash scripts/launch_generation.sh
```

What happens next (unattended):
1. VM starts
2. `generate_and_upload.sh` is copied to the VM
3. Generation begins in the background (`nohup`)
4. Output is uploaded to GCS when done
5. VM shuts itself down

---

### Monitoring a running job

```bash
gcloud compute ssh trainer@oasis-budget-worker \
  --project=ff-ml-project --zone=us-central1-c --tunnel-through-iap \
  --command="tail -f ~/generation.log"
```

---

### Downloading the output

After the VM shuts down, pull the file to your local machine:

```bash
gsutil cp gs://YOUR_BUCKET_NAME/training-data/games_50k_3_.2_.15.jsonl \
  training/data/
```

Then train:
```bash
python3 training/train.py --data training/data/games_50k_3_.2_.15.jsonl \
  --epochs 50 --batch-size 2048
```

---

### Script reference

| Script | Runs on | Purpose |
|--------|---------|---------|
| `scripts/launch_generation.sh` | Local machine | Starts VM, copies job script, triggers run |
| `scripts/generate_and_upload.sh` | GCP VM | Generates games, uploads to GCS, shuts down VM |

---

## VM Commands (GCP — oasis-budget-worker)

Use these when running generation on the cloud VM instead of locally.
VM user: `trainer` | Instance: `oasis-budget-worker` | Project: `ff-ml-project` | Zone: `us-central1-c`

> ⚠️ **Always use Ubuntu** (`ubuntu-2204-lts`) — NOT Debian. The `pypy3` apt package is broken on Debian and requires a manual tarball install.

---

### Step 1 — Create the VM  *(run once, on your local machine)*

```bash
gcloud compute instances create oasis-budget-worker \
  --project=ff-ml-project \
  --zone=us-central1-c \
  --machine-type=c2-standard-16 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --scopes=default
```

---

### Step 2 — SSH into the VM  *(run on your local machine)*

```bash
gcloud compute ssh trainer@oasis-budget-worker \
  --project=ff-ml-project \
  --zone=us-central1-c \
  --tunnel-through-iap
```

---

### Step 3 — First-time VM setup  *(run once, inside the VM)*

```bash
sudo add-apt-repository universe
sudo apt-get update && sudo apt-get install -y pypy3 screen
```

---

### Step 4 — Push training code to VM  *(run on your local machine)*

```bash
gcloud compute scp \
  --project=ff-ml-project \
  --zone=us-central1-c \
  --tunnel-through-iap \
  --recurse ./training trainer@oasis-budget-worker:~/oasis-horse-game/
```

---

### Step 5 — Run generation inside screen  *(run inside the VM)*

```bash
screen -S horse-training

pypy3 training/generate_games.py \
  --n-games 10000 \
  --workers 14 \
  --depth 3 \
  --time-limit 0.2 \
  --epsilon 0.15

# Detach (keep running): Ctrl+A then D
# Reattach later:        screen -r horse-training
```

---

### Step 6 — Verify the file is fully written  *(run inside the VM)*

```bash
wc -l /home/trainer/oasis-horse-game/training/data/games.jsonl
lsof   /home/trainer/oasis-horse-game/training/data/games.jsonl
# lsof should return nothing — if pypy3 still appears, generation is still running
```

---

### Step 7 — Download the data file  *(run on your local machine)*

```bash
gcloud compute scp \
  trainer@oasis-budget-worker:/home/trainer/oasis-horse-game/training/data/games.jsonl \
  ./training/data/games_10000_3_.2_.15.jsonl \
  --project=ff-ml-project \
  --zone=us-central1-c \
  --tunnel-through-iap
```

---

---

## Troubleshooting

**`FileNotFoundError: data/games.jsonl not found`**
→ Run `python training/generate_games.py` first.

**`FileNotFoundError: models/model.pt not found`**
→ Run `python training/train.py` first.

**`[AI] No weights.json found — using heuristic evaluation.`** in browser console
→ Commit and serve `public/model/weights.json`. If running locally, serve via HTTP
  (not file://) — fetch() requires HTTP. Use `python -m http.server 8080` in `public/`.

**Training loss not decreasing**
→ Try more data: `--n-games 50000`. The model is small enough that more data helps more
  than more epochs. Also try `--epochs 50 --lr 5e-4`.

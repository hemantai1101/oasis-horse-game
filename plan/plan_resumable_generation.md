# Plan: Resumable Game Generation

**File to modify:** `training/generate_games.py` only.
**Do NOT touch:** `training/game.py`, `training/train.py`, `public/ai.js`, `public/ai-worker.js`, or any other file.

---

## Goal

Make `generate_games.py` resumable. If the process is killed mid-run (machine shutdown, Ctrl+C, crash), re-running the exact same command should continue from where it left off rather than starting over.

---

## How It Works

Alongside each output `.jsonl` file, maintain a **checkpoint file** (same path, `.checkpoint` extension) that records which game indices have been fully written.

Example:
```
training/data/games_50k_4_.5_.15_v2.jsonl        ← output data
training/data/games_50k_4_.5_.15_v2.checkpoint   ← resume state
```

On startup:
- If **both** the `.jsonl` and `.checkpoint` exist → resume mode: skip completed game indices, open `.jsonl` in append mode
- Otherwise → fresh start: open `.jsonl` in write mode, create empty checkpoint

On completion:
- Delete the `.checkpoint` file (signals a clean finish)

---

## Required Code Changes

### 1. `_worker` must return the game index

**Current:**
```python
def _worker(args_tuple):
    p1_mode, p2_mode, seed, depth, time_limit, epsilon = args_tuple
    random.seed(seed)
    examples = play_game(p1_mode, p2_mode, depth, time_limit, epsilon)
    if examples is None:
        return []
    return examples + apply_180_augmentation(examples)
```

**New:**
```python
def _worker(args_tuple):
    game_idx, p1_mode, p2_mode, seed, depth, time_limit, epsilon = args_tuple
    random.seed(seed)
    examples = play_game(p1_mode, p2_mode, depth, time_limit, epsilon)
    if examples is None:
        return game_idx, []
    return game_idx, examples + apply_180_augmentation(examples)
```

### 2. Work list must include game index

**Current:**
```python
work.append((p1_mode, p2_mode, args.seed + i, args.depth, args.time_limit, args.epsilon))
```

**New:**
```python
work.append((i, p1_mode, p2_mode, args.seed + i, args.depth, args.time_limit, args.epsilon))
```

### 3. Checkpoint helpers (add near top of `main()`)

```python
import json  # already imported

checkpoint_path = out_path.with_suffix('.checkpoint')

def load_checkpoint():
    """Returns set of completed game indices, or empty set if no checkpoint."""
    if checkpoint_path.exists() and out_path.exists():
        with open(checkpoint_path) as f:
            return set(json.load(f))
    return None  # None = fresh start

def save_checkpoint(completed: set):
    with open(checkpoint_path, 'w') as f:
        json.dump(list(completed), f)
```

### 4. Resume logic in `main()`

Replace the current file-open + work-list section with:

```python
completed_indices = load_checkpoint()

if completed_indices is not None:
    # Resume mode
    n_already_done = len(completed_indices)
    work = [w for w in work if w[0] not in completed_indices]
    file_mode = 'a'
    print(f'Resuming: {n_already_done} games already done, {len(work)} remaining.')
else:
    # Fresh start
    completed_indices = set()
    file_mode = 'w'

# Track skipped/written counts from prior run if resuming
# (we can't know exact prior counts so start from 0 for this session)
n_games_total = args.n_games
n_done     = len(completed_indices)   # games already done before this session
n_written  = 0   # records written THIS session (prior session records already in file)
n_skipped  = 0   # skipped THIS session
```

### 5. Main loop — unpack game_idx, update checkpoint

**Current loop body:**
```python
for examples in pool.imap_unordered(_worker, work):
    n_done += 1
    if not examples:
        n_skipped += 1
    else:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
            n_written += 1
    ...
```

**New loop body:**
```python
for game_idx, examples in pool.imap_unordered(_worker, work):
    n_done += 1
    if not examples:
        n_skipped += 1
    else:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
            n_written += 1
        completed_indices.add(game_idx)

    # Save checkpoint every 100 games
    if n_done % 100 == 0:
        f.flush()
        save_checkpoint(completed_indices)
    ...
```

### 6. Cleanup on finish

After the `with open(out_path, file_mode)` block closes, add:

```python
# Clean up checkpoint — generation complete
if checkpoint_path.exists():
    checkpoint_path.unlink()
```

---

## Progress Display

Update the existing print line to show resumed context:

```python
pct = n_done / n_games_total * 100
```

This already works because `n_done` starts at `len(completed_indices)` for resumed runs.

---

## Behaviour Contract

| Scenario | Expected behaviour |
|----------|--------------------|
| Fresh run | Creates `.checkpoint`, writes `.jsonl` from scratch |
| Interrupted mid-run | `.checkpoint` has all games completed before last save (every 100 games) |
| Resume same command | Appends to existing `.jsonl`, skips completed game indices |
| Run completes normally | `.checkpoint` deleted automatically |
| `.jsonl` exists but no `.checkpoint` | Treated as fresh start (overwrites `.jsonl`) |
| `--output` flag changes filename | Fresh start for new filename (different checkpoint path) |

---

## Testing

After implementing, test with:

```bash
# 1. Start a 200-game run
pypy3 training/generate_games.py --n-games 200 --workers 4 --depth 3 --time-limit 0.2 --epsilon 0.15 --output training/data/test_resume.jsonl

# 2. Kill it halfway (Ctrl+C)

# 3. Check checkpoint exists
ls -la training/data/test_resume.*

# 4. Resume — should print "Resuming: X games already done"
pypy3 training/generate_games.py --n-games 200 --workers 4 --depth 3 --time-limit 0.2 --epsilon 0.15 --output training/data/test_resume.jsonl

# 5. Verify total records = expected (should not have duplicates)
python3 -c "
lines = open('training/data/test_resume.jsonl').readlines()
print(f'Total records: {len(lines)}')
"

# 6. Run to completion — checkpoint should be deleted
ls training/data/test_resume.checkpoint  # should: No such file
```

---

## Notes for the Implementing Agent

- The checkpoint flush happens every 100 games — on resume, up to 100 games worth of work may be repeated. This is acceptable (games are deterministic via seed, so repeated games produce identical records).
- Do NOT change the `--seed` arg behaviour. The seed per game is `args.seed + i` where `i` is the global game index — this is what makes games reproducible and resumable.
- The `n_written` and `n_skipped` counters in the progress display reflect THIS session only. The final summary line ("Records written: X") should ideally count total lines in the file, not just this session. Use `out_path` line count for the final summary if possible.
- Keep `imap_unordered` — do not switch to `imap`. The out-of-order completion is fine since we track by index.
- Do not add a `--resume` flag. Resume should be automatic based on checkpoint file presence.

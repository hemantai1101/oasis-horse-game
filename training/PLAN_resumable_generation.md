# Plan: Resumable Game Generation

**Task for parallel agent:** Add resume/continue support to `generate_games.py` so that if
the machine shuts down mid-run, generation can continue from where it left off.

**Do not start until Run 006 export + test is done** — `generate_games.py` is stable after
the 105-feature changes and this plan is based on its current state.

---

## Problem

Currently `generate_games.py`:
- Opens output file in `'w'` mode — overwrites everything on restart
- `_worker()` returns only `examples` — no way to know which game index completed
- No state is saved between runs

## Solution: Checkpoint File

Alongside `games_50k_4_.5_.15_v2.jsonl` keep a `games_50k_4_.5_.15_v2.checkpoint.json`
that tracks which game indices have been written.

### Resume flow
```
Start
  └─ checkpoint file exists AND output file exists?
       Yes → load completed set of game indices
             filter work list to only remaining games
             open output file in 'a' (append) mode
             print "Resuming from N/M completed"
       No  → fresh start, open output file in 'w' mode

Run imap_unordered on remaining work
  └─ each result: write records, save index to checkpoint

Finish
  └─ delete checkpoint file (signals clean completion)
```

---

## Exact Code Changes

### 1. `_worker` — return game index alongside examples

**File:** `training/generate_games.py`

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

**Change to:**
```python
def _worker(args_tuple):
    game_index, p1_mode, p2_mode, seed, depth, time_limit, epsilon = args_tuple
    random.seed(seed)
    examples = play_game(p1_mode, p2_mode, depth, time_limit, epsilon)
    if examples is None:
        return game_index, []
    return game_index, examples + apply_180_augmentation(examples)
```

### 2. Work list — prepend game index

**Current:**
```python
work.append((p1_mode, p2_mode, args.seed + i, args.depth, args.time_limit, args.epsilon))
```

**Change to:**
```python
work.append((i, p1_mode, p2_mode, args.seed + i, args.depth, args.time_limit, args.epsilon))
```

### 3. Add checkpoint helpers (new functions, add before `main()`)

```python
def load_checkpoint(checkpoint_path):
    """Returns set of completed game indices, or empty set if no checkpoint."""
    if not checkpoint_path.exists():
        return set()
    with open(checkpoint_path) as f:
        return set(json.load(f))


def save_checkpoint(checkpoint_path, completed):
    """Atomically save the set of completed game indices."""
    tmp = checkpoint_path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(list(completed), f)
    tmp.replace(checkpoint_path)  # atomic on POSIX
```

### 4. `main()` — resume logic + checkpoint save per game

Replace the section from `# Build output path...` through the `with open(out_path, 'w')` block:

```python
# Build output path relative to this script's directory
script_dir  = Path(__file__).parent
out_path    = Path(args.output) if args.output else script_dir / 'data' / 'games.jsonl'
out_path.parent.mkdir(parents=True, exist_ok=True)
ckpt_path   = out_path.with_suffix('.checkpoint.json')

# Resume detection
completed = load_checkpoint(ckpt_path)
resuming  = len(completed) > 0 and out_path.exists()
file_mode = 'a' if resuming else 'w'

if resuming:
    print(f'Resuming: {len(completed)}/{n_games} games already done → {out_path}')
else:
    print(f'Generating {n_games} games → {out_path}')

# Filter work list to only remaining games
work = [(i, p1, p2, args.seed + i, args.depth, args.time_limit, args.epsilon)
        for i, (p1, p2) in enumerate(
            random.choices(modes, weights=weights, k=1)[0]  # ← keep existing selection logic
        )
        if i not in completed]
```

⚠️ The work-list build loop needs to be rewritten to keep existing random mode selection AND
   filter out completed indices. Here is the correct version:

```python
random.seed(args.seed)  # reset seed so selection is reproducible
work = []
for i in range(args.n_games):
    p1_mode, p2_mode = random.choices(modes, weights=weights, k=1)[0]
    if i not in completed:
        work.append((i, p1_mode, p2_mode, args.seed + i, args.depth, args.time_limit, args.epsilon))
```

Then inside the `imap_unordered` loop, replace the existing result handling:

```python
# OLD:
for examples in pool.imap_unordered(_worker, work):
    n_done += 1
    if not examples:
        n_skipped += 1
    else:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
            n_written += 1

# NEW:
for game_index, examples in pool.imap_unordered(_worker, work):
    n_done += 1
    if not examples:
        n_skipped += 1
    else:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
            n_written += 1
    completed.add(game_index)
    save_checkpoint(ckpt_path, completed)
    # ... rest of progress printing unchanged
```

And at the very end after the loop, delete the checkpoint:

```python
# Clean up checkpoint on successful completion
if ckpt_path.exists():
    ckpt_path.unlink()
    print(f'  Checkpoint deleted (generation complete)')
```

### 5. Open file with correct mode

```python
# Change:
with open(out_path, 'w') as f:
# To:
with open(out_path, file_mode) as f:
```

---

## Testing Plan

1. Run with `--n-games 20 --workers 3` — let it complete 8 games, then Ctrl+C
2. Check that `.checkpoint.json` exists with ~8 indices
3. Re-run same command — should print "Resuming: 8/20 games already done"
4. Should complete and produce the same total record count as a fresh 20-game run
5. Verify checkpoint file is deleted on clean finish

---

## Files Changed

| File | Change |
|------|--------|
| `training/generate_games.py` | Add `load_checkpoint`, `save_checkpoint`; update `_worker`, work list build, `imap_unordered` loop |

No other files need changes. `game.py`, `train.py`, `ai-worker.js` are unaffected.

---

## Notes

- The checkpoint file is ~4 bytes per completed game index (JSON int array). For 50k games: ~200KB max. Negligible.
- If the machine dies mid-write for a single game's records, that game index is NOT in the checkpoint → it gets replayed on resume. At most one game is replayed — safe duplicate risk is low.
- Atomic write (`tmp → replace`) prevents a corrupt checkpoint if the machine dies during checkpoint save.
- Works correctly with `imap_unordered` — we track by game index, not line count.

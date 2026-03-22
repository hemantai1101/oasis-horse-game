# Training Run Log

Master summary of all training runs. **Full details in `training/runs/run_NNN.md`.**

---

## Quick Reference

| Run | Date | Data | Features | Architecture | Val Loss | Status | Key Change |
|-----|------|------|----------|-------------|----------|--------|------------|
| [001](runs/run_001.md) | 2026-03-13 | 10k games, depth=3 | 41 | 41→128→64→1 | 0.60162 | ✅ Exported & tested | Baseline |
| [002](runs/run_002.md) | 2026-03-14 | 50k games, depth=3 | 41 | 41→128→64→1 | 0.61344 | ❌ Not converged | 5× more data |
| [003](runs/run_003.md) | 2026-03-14 | 50k games, depth=3 | 41 | 41→128→64→1 | 0.60809 | ✅ Exported & tested | 100 epochs (converged) |
| [004](runs/run_004.md) | 2026-03-15 | 50k games, depth=4 | 41 | 41→128→64→1 | 0.70706 | ❌ Superseded | depth=4 teacher |
| [005](runs/run_005.md) | 2026-03-15 | 50k games, depth=4 | 41 | 41→256→128→1 | 0.69068 | ✅ Exported & tested | 3× wider model |
| [006](runs/run_006.md) | 2026-03-17 | 50k games, depth=4 | **105** | 105→256→128→1 | **0.62254** | ✅ Exported & tested | **Feature engineering** |
| [007](runs/run_007.md) | 2026-03-18 | 50k games, depth=4 | **110** | 110→256→128→1 | 0.63295 | ❌ Regression | path_threat graded (backfired) |
| [008](runs/run_008.md) | 2026-03-19 | 15k+50k games, depth=4 | **110** | 110→256→128→1 | 0.60674 (15k) / 0.62827 (50k) | ❌ Underfitted at 100 epochs | Revert path_clear + fix normalizations |
| [009](runs/run_009.md) | 2026-03-19 | 50k games, depth=4 | **110** | 110→256→128→1 | 0.62504 | ✅ Exported & tested | 200 epochs — confirmed depth=4 ceiling |
| [010](runs/run_010.md) | 2026-03-21 | 15k games, depth=5 | **110** | 110→256→128→1 | **0.56910** | ✅ Best model | **depth=5 teacher — biggest jump since Run 006** |

---

## Progression Summary

```
Run 001  Baseline: depth-3, 10k, narrow    → 0.602  LR exhausted at epoch 48
Run 002  More data: depth-3, 50k, narrow   → 0.613  Not converged (50 epochs not enough)
Run 003  Converge:  depth-3, 50k, narrow   → 0.608  ← Ceiling for depth-3 + 41 features
Run 004  Stronger teacher: depth-4, narrow → 0.707  Model too small for depth-4 complexity
Run 005  Wider model: depth-4, wide        → 0.691  ← Ceiling for 41 raw position features
Run 006  Feature engineering: depth-4, 105 → 0.622  ← Biggest jump yet (+0.069 vs Run 005)
Run 007  Explicit threats:   depth-4, 110 → 0.633  ← REGRESSION: graded path_threat hurt model
Run 008  Fix normalizations: depth-4, 110 → 0.607* ← Normalization fix confirmed on 15k; 50k underfitted at 100 epochs
Run 009  Full convergence:   depth-4, 110 → 0.625  ← Confirmed depth=4 data ceiling
Run 010  Depth-5 teacher:   depth-5, 110 → 0.569  ← Biggest jump since Run 006 (+0.053)
```

---

## Key Lessons

| Lesson | Evidence |
|--------|---------|
| More data alone doesn't help after depth-3 ceiling | Run 001→002→003: val loss plateaued ~0.608 |
| Depth=4 data is harder to fit — val loss is not comparable across depths | Run 003 (0.608) vs Run 004 (0.707) — different difficulty |
| Model capacity matters but has diminishing returns | Run 004→005: 3× params only bought 0.016 improvement |
| 41 raw features cannot encode game-critical concepts | Run 005 in-game test: AI can't beat human, no tactical awareness |
| Val loss is an imperfect proxy for playing strength | Best measure is human vs AI gameplay |
| Depth=4 training data has a hard ceiling ~0.624 | Runs 008/009 both converged there regardless of epochs or data volume |
| Stronger teacher depth dramatically improves leaf evaluations | Run 010: depth=5 data → 0.569 vs depth=4 data → 0.625 with same architecture |
| 20% skip rate at depth=5 is acceptable | Surviving games are high quality; 12k depth=5 >> 50k depth=4 |

---

## Feature Versions

| Version | Feature Count | Description |
|---------|--------------|-------------|
| v1 (Runs 001–005) | 41 | Raw (col, row) per horse + player indicator |
| v2 (Run 006) | 105 | + dist to center, on-axis flag, path-clear flag, 4 backstop cells |
| v3 (Run 007) | 110 | + path_threat graded (0–1.0), winning_threats/10, at-home/10, blocking/10 — all /10 normalization caused weak gradients |
| v4 (Run 008+) | 110 | + path_clear binary (reverted), winning_threats/2, at-home/10, blocking/5 — corrected normalizations |

---

## Current Best Model

**Run 010** — `training/models/model_run010.pt`
- Architecture: 110→256→128→1
- Val loss: **0.56910** ← best val loss across all runs
- Data: 15k games, depth=5 teacher, 2.0s time limit
- In-game: pending export & test

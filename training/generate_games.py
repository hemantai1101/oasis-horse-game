"""
generate_games.py — Self-play data generator for the Squirrel Oasis training pipeline.

Plays N games between minimax AIs of varying strength to build a diverse dataset.
Records every board position encountered, labelled with the eventual game outcome.
Applies 180° board rotation as data augmentation (doubles the dataset size).

Output: training/data/games.jsonl
Each line is a JSON object: {"features": [21 floats], "label": 1.0 or -1.0}
  label = +1.0 if the player to move at that position won
  label = -1.0 if they lost

Usage:
  python generate_games.py                  # 10,000 games (default)
  python generate_games.py --n-games 100    # quick smoke test
  python generate_games.py --n-games 50000  # large dataset for stage-2 training
"""

import argparse
import json
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path

from game import (
    make_initial_horses, horses_to_board, apply_move,
    is_won, state_to_features, augment_horses_180, CENTER,
)
from minimax import get_best_move, get_random_move


# ===== Game Mix =====
# Each tuple: (p1_mode, p2_mode)
#   'depth3' = minimax depth 3   (strongest, ~expert level)
#   'depth2' = minimax depth 2   (slightly weaker, more mistakes)
#   'random' = uniform random    (very weak, creates diverse positions)
GAME_MIX = [
    ('depth3', 'depth3', 0.60),   # 60% — high-quality expert games
    ('depth2', 'depth3', 0.15),   # 15% — P1 weaker
    ('depth3', 'depth2', 0.15),   # 15% — P2 weaker
    ('random', 'depth3', 0.05),   # 5%  — P1 random
    ('depth3', 'random', 0.05),   # 5%  — P2 random
]

MAX_MOVES   = 400   # safety cap — skip games that drag on too long


# ===== Single Game =====

def play_game(p1_mode, p2_mode, depth, time_limit, epsilon):
    """
    Play one game. Returns a list of (features, label) pairs.
    label = +1.0 if the player to move eventually won, -1.0 if lost.
    Returns None if the game hits the move cap (treated as useless data).
    """
    horses, board = make_initial_horses()
    current_player = 1
    history = []   # list of (features, player_to_move)

    modes  = {1: p1_mode, 2: p2_mode}

    for _ in range(MAX_MOVES):
        # Record position BEFORE the move
        feats = state_to_features(horses, current_player)
        history.append((feats, current_player))

        # Choose a move
        mode = modes[current_player]
        if mode == 'random':
            result = get_random_move(horses, board, current_player)
        else:
            result = get_best_move(horses, board, current_player,
                                   depth=depth, time_limit=time_limit, epsilon=epsilon)

        if result is None:
            return None  # no legal moves — skip game

        horse_id, move = result
        horses, board = apply_move(horses, board, horse_id, move)

        if is_won(board):
            winner = current_player
            # Build labelled examples
            examples = []
            for feats, player in history:
                label = 1.0 if player == winner else -1.0
                examples.append({'features': feats, 'label': label})
            return examples

        current_player = 3 - current_player

    return None  # hit move cap


def apply_180_augmentation(examples):
    """
    Apply 180° board rotation to a list of examples.
    Rotation: col → 12-col, row → 12-row.
    This preserves player-corner assignments (TL↔BR, TR↔BL stay with the same owner).
    Returns the augmented examples (original are NOT included — caller combines them).
    """
    augmented = []
    for ex in examples:
        feats = ex['features']
        # features layout: [p1: 10 vals][p2: 10 vals][player: 1 val]
        # each pair (col_norm, row_norm) where col_norm = (col-1)/10
        # rotation: col → (12-col), so col_norm' = (12 - (col_norm*10+1) - 1)/10
        #         = (10 - col_norm*10)/10 = 1 - col_norm
        # same for row_norm
        new_feats = list(feats)
        for i in range(0, 20, 2):      # 10 pairs in positions [0..19]
            new_feats[i]   = 1.0 - feats[i]    # col
            new_feats[i+1] = 1.0 - feats[i+1]  # row
        # player indicator stays the same (rotation doesn't change whose turn it is)
        augmented.append({'features': new_feats, 'label': ex['label']})
    return augmented


# ===== Main =====

def _worker(args_tuple):
    """Worker function for multiprocessing — plays one game and returns examples."""
    p1_mode, p2_mode, seed, depth, time_limit, epsilon = args_tuple
    random.seed(seed)
    examples = play_game(p1_mode, p2_mode, depth, time_limit, epsilon)
    if examples is None:
        return []
    return examples + apply_180_augmentation(examples)


def main():
    parser = argparse.ArgumentParser(description='Generate self-play training data')
    parser.add_argument('--n-games', type=int, default=10_000,
                        help='Number of games to play (default: 10000)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: data/games.jsonl)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--workers', type=int, default=max(1, multiprocessing.cpu_count() - 1),
                        help='Parallel worker processes (default: cpu_count-1)')
    parser.add_argument('--depth', type=int, default=5, help='Minimax search depth')
    parser.add_argument('--time-limit', type=float, default=1.0, help='Time limit per move in seconds')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon for epsilon-greedy move selection')
    args = parser.parse_args()

    random.seed(args.seed)

    # Build output path relative to this script's directory
    script_dir = Path(__file__).parent
    out_path   = Path(args.output) if args.output else script_dir / 'data' / 'games.jsonl'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    modes   = [(p1, p2) for p1, p2, _ in GAME_MIX]
    weights = [w for _, _, w in GAME_MIX]

    # Build work list: (p1_mode, p2_mode, per_game_seed, depth, time_limit, epsilon)
    work = []
    for i in range(args.n_games):
        p1_mode, p2_mode = random.choices(modes, weights=weights, k=1)[0]
        work.append((p1_mode, p2_mode, args.seed + i, args.depth, args.time_limit, args.epsilon))

    n_games    = args.n_games
    n_written  = 0
    n_skipped  = 0
    n_done     = 0
    start_time = time.time()

    print(f'Generating {n_games} games → {out_path}')
    print(f'Using {args.workers} parallel workers')
    print(f'Depth: {args.depth}, Time Limit: {args.time_limit}s, Epsilon: {args.epsilon}')
    print('(Ctrl+C to stop early — partial output is still valid)\n')

    with open(out_path, 'w') as f:
        with multiprocessing.Pool(processes=args.workers) as pool:
            for examples in pool.imap_unordered(_worker, work):
                n_done += 1
                if not examples:
                    n_skipped += 1
                else:
                    for ex in examples:
                        f.write(json.dumps(ex) + '\n')
                        n_written += 1

                elapsed   = time.time() - start_time
                rate      = n_done / elapsed if elapsed > 0 else 0
                remaining = (n_games - n_done) / rate if rate > 0 else 0
                pct       = n_done / n_games * 100
                skip_pct  = n_skipped / n_done * 100 if n_done > 0 else 0
                print(
                    f'  [{pct:5.1f}%] game {n_done}/{n_games} | '
                    f'skip {skip_pct:.0f}% | '
                    f'{n_written} records | '
                    f'{rate:.2f} games/s | '
                    f'ETA {remaining:.0f}s',
                    end='\r', flush=True
                )

    elapsed = time.time() - start_time
    print(f'\n\nDone in {elapsed:.1f}s')
    print(f'  Games played:  {n_games - n_skipped} / {n_games} (skipped {n_skipped})')
    print(f'  Records written: {n_written} (including 2× augmentation)')
    print(f'  Output: {out_path}')


if __name__ == '__main__':
    main()

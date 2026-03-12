"""
minimax.py — Python port of the Squirrel Oasis AI (mirrors ai.js).

The evaluation and search logic here must match the JS AI so that
self-play games represent the same quality of play as the browser AI.

Key structures mirrored from ai.js:
  - APPROACH_CONFIGS (4 backstop cells)
  - scoreBackstopControl → score_backstop_control
  - evaluate → evaluate_position
  - generateOrderedMoves → generate_ordered_moves
  - minimax → minimax_search
  - getAIMove → get_best_move
"""

import time
import math
import random
from game import (
    CENTER, BOARD_SIZE, valid_moves, apply_move,
    horses_to_board, is_won,
)


# ===== Approach configs — mirrors APPROACH_CONFIGS in ai.js =====
#
# Each entry: {bs: (col,row), filter: callable(horse) -> bool}
# filter returns True if the horse is a valid "attacker" for this axis.

def _make_approach_configs():
    cx, cy = CENTER
    return [
        # Attacker above center, slides DOWN, backstop at (6,7)
        {'bs': (cx, cy+1), 'filter': lambda h: h['col'] == cx and h['row'] < cy},
        # Attacker below center, slides UP, backstop at (6,5)
        {'bs': (cx, cy-1), 'filter': lambda h: h['col'] == cx and h['row'] > cy},
        # Attacker left of center, slides RIGHT, backstop at (7,6)
        {'bs': (cx+1, cy), 'filter': lambda h: h['row'] == cy and h['col'] < cx},
        # Attacker right of center, slides LEFT, backstop at (5,6)
        {'bs': (cx-1, cy), 'filter': lambda h: h['row'] == cy and h['col'] > cx},
    ]

APPROACH_CONFIGS = _make_approach_configs()


# ===== Threat Analysis =====

def path_clear(horse, board):
    """
    True if every cell strictly between horse and CENTER (along row or col) is empty.
    Mirrors pathClear() in ai.js.
    """
    cx, cy = CENTER
    hx, hy = horse['col'], horse['row']
    dc = int(math.copysign(1, cx - hx)) if cx != hx else 0
    dr = int(math.copysign(1, cy - hy)) if cy != hy else 0
    if dc != 0 and dr != 0:
        return False  # not on same row or column
    c, r = hx + dc, hy + dr
    while (c, r) != CENTER:
        if (c, r) in board:
            return False
        c += dc
        r += dr
    return True


def score_backstop_control(horses, board, ai_player):
    """Mirrors scoreBackstopControl() in ai.js."""
    opp = 3 - ai_player
    score = 0

    for cfg in APPROACH_CONFIGS:
        bs_pos = cfg['bs']
        occupant = board.get(bs_pos)
        if occupant is None:
            continue

        for h in horses:
            if not cfg['filter'](h):
                continue
            if not path_clear(h, board):
                continue

            # A live attack exists on this axis
            h_owner  = h['owner']
            bs_owner = occupant['owner']

            if h_owner == ai_player and bs_owner == ai_player:
                score += 1500   # AI has a winning setup
            elif h_owner == opp and bs_owner == opp:
                score -= 2000   # Opponent is about to win
            elif h_owner == opp and bs_owner == ai_player:
                score -= 800    # AI piece is opponent's backstop (bad)
            elif h_owner == ai_player and bs_owner == opp:
                score += 500    # Opponent piece acts as AI's backstop (opportunistic)

    return score


def evaluate_position(horses, board, ai_player):
    """
    Mirrors evaluate() in ai.js.
    Returns a score from the AI player's perspective (higher = better for AI).
    """
    score = score_backstop_control(horses, board, ai_player)

    # Near-win bonus: aligned + clear path, but no backstop yet
    for cfg in APPROACH_CONFIGS:
        if board.get(cfg['bs']) is not None:
            continue  # already handled in backstop control
        for h in horses:
            if not cfg['filter'](h):
                continue
            if not path_clear(h, board):
                continue
            score += 300 if h['owner'] == ai_player else -400

    # General positioning — closeness to center, penalise off-axis forest clustering
    cx, cy = CENTER
    for h in horses:
        dist = abs(h['col'] - cx) + abs(h['row'] - cy)
        on_axis  = h['col'] == cx or h['row'] == cy
        in_forest = 0 < dist <= 2
        pos = (10 - dist) * 8
        # Only penalise off-axis pieces in the forest — on-axis pieces are valuable
        if in_forest and not on_axis:
            pos *= 0.3
        if h['owner'] == ai_player:
            score += pos
        else:
            score -= pos

    # Fix 1: axis-alignment bonus — reward pieces on col=6 or row=6
    for h in horses:
        if h['col'] != cx and h['row'] != cy:
            continue
        if h['owner'] == ai_player:
            score += 60
        else:
            score -= 60

    # Fix 2: preemptive backstop bonus — reward occupying a backstop cell
    # even before an aligned attacker exists
    backstop_cells = {(cx, cy-1), (cx, cy+1), (cx-1, cy), (cx+1, cy)}
    for h in horses:
        if (h['col'], h['row']) not in backstop_cells:
            continue
        if h['owner'] == ai_player:
            score += 80
        else:
            score -= 80

    return score


# ===== Move Ordering =====

def generate_ordered_moves(horses, board, player, ai_player):
    """
    Mirrors generateOrderedMoves() in ai.js.
    Returns list of (horse_id, move_dict), sorted best-first.
    """
    opp = 3 - player
    moves = []
    for h in horses:
        if h['owner'] != player:
            continue
        for move in valid_moves(h, board):
            moves.append((h['id'], move, h))

    def priority(item):
        _, move, h = item
        col, row = move['col'], move['row']

        # 1. Immediate win
        if (col, row) == CENTER:
            return 1_000_000_000

        pri = 0

        # 2. Creating a winning backstop for an aligned friendly attacker
        for cfg in APPROACH_CONFIGS:
            if (col, row) == cfg['bs']:
                for ah in horses:
                    if ah['owner'] != player or not cfg['filter'](ah):
                        continue
                    if path_clear(ah, board):
                        pri = max(pri, 8000)

        # 3. Blocking opponent's aligned attacker
        cx, cy = CENTER
        for cfg in APPROACH_CONFIGS:
            for ah in horses:
                if ah['owner'] != opp or not cfg['filter'](ah):
                    continue
                if not path_clear(ah, board):
                    continue
                dc = int(math.copysign(1, cx - ah['col'])) if cx != ah['col'] else 0
                dr = int(math.copysign(1, cy - ah['row'])) if cy != ah['row'] else 0
                c, r = ah['col'] + dc, ah['row'] + dr
                while (c, r) != CENTER:
                    if col == c and row == r:
                        pri = max(pri, 5000)
                    c += dc
                    r += dr

        # 4. Fallback: proximity to center
        if pri == 0:
            dist = abs(col - CENTER[0]) + abs(row - CENTER[1])
            pri = 10 - dist

        return pri

    moves.sort(key=priority, reverse=True)
    return [(h_id, move) for h_id, move, _ in moves]


# ===== Minimax =====

def minimax_search(horses, board, depth, alpha, beta, maximizing, ai_player, deadline):
    """
    Mirrors minimax() in ai.js.
    Returns a score. Uses alpha-beta pruning.
    deadline: time.time() value after which we return early.
    """
    if time.time() > deadline:
        return alpha if maximizing else beta

    # Check terminal: any piece at center?
    if CENTER in board:
        winner = board[CENTER]['owner']
        return +10000 if winner == ai_player else -10000

    if depth == 0:
        return evaluate_position(horses, board, ai_player)

    player = ai_player if maximizing else (3 - ai_player)
    moves  = generate_ordered_moves(horses, board, player, ai_player)

    if not moves:
        return evaluate_position(horses, board, ai_player)

    if maximizing:
        best = -math.inf
        for horse_id, move in moves:
            new_horses, new_board = apply_move(horses, board, horse_id, move)
            val = minimax_search(new_horses, new_board, depth - 1, alpha, beta, False, ai_player, deadline)
            best = max(best, val)
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return best
    else:
        best = +math.inf
        for horse_id, move in moves:
            new_horses, new_board = apply_move(horses, board, horse_id, move)
            val = minimax_search(new_horses, new_board, depth - 1, alpha, beta, True, ai_player, deadline)
            best = min(best, val)
            beta = min(beta, val)
            if beta <= alpha:
                break
        return best


# ===== Root Search =====

def get_best_move(horses, board, player, depth=5, time_limit=2.0, randomise_equal=True):
    """
    Returns (horse_id, move_dict) for the best move at the given depth.
    Mirrors getAIMove() in ai.js.

    randomise_equal: shuffle equally-scored moves to add variety to self-play games.
    """
    deadline = time.time() + time_limit
    moves = generate_ordered_moves(horses, board, player, player)

    if not moves:
        return None

    # Immediate win check
    first_horse_id, first_move = moves[0]
    if (first_move['col'], first_move['row']) == CENTER:
        return first_horse_id, first_move

    best_val  = -math.inf
    best_move = moves[0]

    for horse_id, move in moves:
        new_horses, new_board = apply_move(horses, board, horse_id, move)
        # After AI moves, opponent maximises from their own perspective
        val = minimax_search(
            new_horses, new_board,
            depth - 1,
            -math.inf, +math.inf,
            False,           # opponent minimises AI score
            player,          # ai_player stays the same for the full search
            deadline,
        )
        if val > best_val:
            best_val  = val
            best_move = (horse_id, move)

    return best_move


def get_random_move(horses, board, player):
    """Return a uniformly random legal move for the given player."""
    moves = []
    for h in horses:
        if h['owner'] != player:
            continue
        for move in valid_moves(h, board):
            moves.append((h['id'], move))
    if not moves:
        return None
    return random.choice(moves)

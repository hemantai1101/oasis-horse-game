"""
minimax.py — Python port of the Squirrel Oasis AI (mirrors ai.js).

The evaluation and search logic here must match the JS AI so that
self-play games represent the same quality of play as the browser AI.

Key structures mirrored from ai.js:
  - APPROACH_CONFIGS (4 backstop cells)
  - evaluate → evaluate
  - generateOrderedMoves → generate_ordered_moves
  - minimax → minimax
  - getAIMove → get_best_move
"""

import time
import math
import random
from game import (
    CENTER, BOARD_SIZE, valid_moves, apply_move,
    horses_to_board, is_won, state_to_features
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

def count_mobility(horse, board):
    return len(valid_moves(horse, board))

# ===== EVALUATION (complete rewrite to match ai.js) =====

def evaluate(snap, ai_player):
    """
    Mirrors evaluate() in ai.js.
    Returns a score from the ai_player's perspective.
    snap is a dict with 'horses' and 'board' (and 'currentPlayer' which we don't need here)
    """
    ai = ai_player
    opp = 3 - ai
    score = 0

    horses = snap['horses']
    board = snap['board']

    ai_horses = [h for h in horses if h['owner'] == ai]
    opp_horses = [h for h in horses if h['owner'] == opp]

    # 1. TERMINAL THREAT SCORING
    for cfg in APPROACH_CONFIGS:
        bs_pos = cfg['bs']
        occupant = board.get(bs_pos)

        if occupant:
            for h in horses:
                if not cfg['filter'](h): continue
                if not path_clear(h, board): continue

                if h['owner'] == ai and occupant['owner'] == ai:
                    score += 3000
                elif h['owner'] == opp and occupant['owner'] == opp:
                    score -= 4000
                elif h['owner'] == opp and occupant['owner'] == ai:
                    score -= 1200
                elif h['owner'] == ai and occupant['owner'] == opp:
                    score += 800
        else:
            for h in horses:
                if not cfg['filter'](h): continue
                if not path_clear(h, board): continue
                score += 400 if h['owner'] == ai else -600

    # 2. PIECE DEVELOPMENT & POSITIONING
    cx, cy = CENTER
    for h in horses:
        dist = abs(h['col'] - cx) + abs(h['row'] - cy)
        on_axis = h['col'] == cx or h['row'] == cy
        sign = 1 if h['owner'] == ai else -1

        score += sign * (10 - dist) * 15

        if on_axis:
            score += sign * 100
            if dist <= 3:
                score += sign * 80

        in_corner = (
            (h['col'] <= 3 and h['row'] <= 3) or
            (h['col'] <= 3 and h['row'] >= 9) or
            (h['col'] >= 9 and h['row'] <= 3) or
            (h['col'] >= 9 and h['row'] >= 9)
        )
        if in_corner:
            score -= sign * 60

        if (h['col'], h['row']) in [(6,5), (6,7), (5,6), (7,6)]:
            score += sign * 120

    # 3. MOBILITY
    ai_mobility = sum(count_mobility(h, board) for h in ai_horses)
    opp_mobility = sum(count_mobility(h, board) for h in opp_horses)
    score += (ai_mobility - opp_mobility) * 8

    # 4. PIECE SPREAD / DEVELOPMENT DIVERSITY
    ai_developed = 0
    for h in ai_horses:
        in_start_corner = (h['col'] >= 9 and h['row'] <= 3) or (h['col'] <= 3 and h['row'] >= 9) # Assuming AI is P2 usually, but let's make it robust
        # Actually in ai.js it's hardcoded based on start positions. Let's replicate.
        if h['owner'] == 2:
            in_start_corner = (h['col'] >= 9 and h['row'] <= 3) or (h['col'] <= 3 and h['row'] >= 9)
        else:
            in_start_corner = (h['col'] <= 3 and h['row'] <= 3) or (h['col'] >= 9 and h['row'] >= 9)

        if not in_start_corner: ai_developed += 1
    score += ai_developed * 25 if ai == 2 else ai_developed * -25 # Wait, the sign is handled differently in JS. Let's fix this based on JS logic.

    # Re-evaluating Section 4 to perfectly match JS:
    # In JS:
    # aiDeveloped = pieces NOT in P2 corners (if AI is P2)
    # oppDeveloped = pieces NOT in P1 corners (if Opp is P1)
    # This assumes AI is ALWAYS Player 2 in JS. In python we might play as P1 or P2.
    # Let's generalize based on owner.
    ai_dev = 0
    opp_dev = 0
    for h in ai_horses:
        if h['owner'] == 1:
            in_start = (h['col'] <= 3 and h['row'] <= 3) or (h['col'] >= 9 and h['row'] >= 9)
        else:
            in_start = (h['col'] >= 9 and h['row'] <= 3) or (h['col'] <= 3 and h['row'] >= 9)
        if not in_start: ai_dev += 1

    for h in opp_horses:
        if h['owner'] == 1:
            in_start = (h['col'] <= 3 and h['row'] <= 3) or (h['col'] >= 9 and h['row'] >= 9)
        else:
            in_start = (h['col'] >= 9 and h['row'] <= 3) or (h['col'] <= 3 and h['row'] >= 9)
        if not in_start: opp_dev += 1

    score += ai_dev * 25
    score -= opp_dev * 25

    # 5. AXIS CONTROL
    ai_on_axis = sum(1 for h in ai_horses if h['col'] == cx or h['row'] == cy)
    opp_on_axis = sum(1 for h in opp_horses if h['col'] == cx or h['row'] == cy)
    score += (ai_on_axis - opp_on_axis) * 40

    # 6. BLOCKING AWARENESS
    for cfg in APPROACH_CONFIGS:
        for h in opp_horses:
            if not cfg['filter'](h): continue
            if not path_clear(h, board): continue

            dc = int(math.copysign(1, cx - h['col'])) if cx != h['col'] else 0
            dr = int(math.copysign(1, cy - h['row'])) if cy != h['row'] else 0
            c, r = h['col'] + dc, h['row'] + dr
            path_cells = []
            while (c, r) != CENTER:
                path_cells.append((c, r))
                c += dc; r += dr

            can_block = False
            for ah in ai_horses:
                moves = valid_moves(ah, board)
                for m in moves:
                    if (m['col'], m['row']) in path_cells:
                        can_block = True
                        break
                if can_block: break

            if not can_block:
                score -= 500

    return score


# ===== Move Ordering =====

def generate_ordered_moves(snap):
    """
    Mirrors generateOrderedMoves() in ai.js.
    """
    player = snap['currentPlayer']
    opp = 3 - player
    moves = []
    horses = snap['horses']
    board = snap['board']

    for h in horses:
        if h['owner'] != player:
            continue
        for move in valid_moves(h, board):
            moves.append({'horseId': h['id'], 'move': move})

    for m in moves:
        move = m['move']
        horse_id = m['horseId']
        col, row = move['col'], move['row']
        pri = 0

        # 1. Immediate win
        if (col, row) == CENTER:
            pri = 1_000_000_000
        else:
            # 2. Creating backstop
            for cfg in APPROACH_CONFIGS:
                if (col, row) == cfg['bs']:
                    for h in horses:
                        if h['owner'] != player or not cfg['filter'](h): continue
                        if path_clear(h, board):
                            pri = max(pri, 8000)

            # 3. Moving aligned attacker
            for cfg in APPROACH_CONFIGS:
                horse = next((h for h in horses if h['id'] == horse_id), None)
                if not horse or not cfg['filter'](horse): continue
                if not path_clear(horse, board): continue
                occupant = board.get(cfg['bs'])
                if occupant and occupant['owner'] == player:
                    pri = max(pri, 7000)

            # 4. Blocking
            for cfg in APPROACH_CONFIGS:
                for h in horses:
                    if h['owner'] != opp or not cfg['filter'](h): continue
                    if not path_clear(h, board): continue
                    cx, cy = CENTER
                    dc = int(math.copysign(1, cx - h['col'])) if cx != h['col'] else 0
                    dr = int(math.copysign(1, cy - h['row'])) if cy != h['row'] else 0
                    c, r = h['col'] + dc, h['row'] + dr
                    while (c, r) != CENTER:
                        if col == c and row == r:
                            pri = max(pri, 6000)
                        c += dc; r += dr

            # 5. Axis
            if pri == 0 and (col == CENTER[0] or row == CENTER[1]):
                dist = abs(col - CENTER[0]) + abs(row - CENTER[1])
                pri = max(pri, 100 + (10 - dist) * 10)

            # 6. Fallback
            if pri == 0:
                dist = abs(col - CENTER[0]) + abs(row - CENTER[1])
                pri = 10 - dist

        m['priority'] = pri

    moves.sort(key=lambda x: x['priority'], reverse=True)
    return moves


# ===== Minimax =====

def minimax(snap, depth, alpha, beta, maximizing, ai_player, deadline):
    """
    Mirrors minimax() in ai.js.
    """
    if time.time() > deadline:
        return alpha if maximizing else beta

    # Terminal check
    if snap.get('lastMoveToCenter'):
        winner = 3 - snap['currentPlayer']
        return +50000 if winner == ai_player else -50000

    if depth == 0:
        return evaluate(snap, ai_player)

    moves = generate_ordered_moves(snap)
    if not moves:
        return evaluate(snap, ai_player)

    if maximizing:
        best = -math.inf
        for m in moves:
            new_horses, new_board = apply_move(snap['horses'], snap['board'], m['horseId'], m['move'])
            child_snap = {
                'horses': new_horses,
                'board': new_board,
                'currentPlayer': 3 - snap['currentPlayer'],
                'lastMoveToCenter': m['move']['col'] == CENTER[0] and m['move']['row'] == CENTER[1]
            }
            val = minimax(child_snap, depth - 1, alpha, beta, False, ai_player, deadline)
            best = max(best, val)
            alpha = max(alpha, val)
            if alpha >= beta: break
        return best
    else:
        best = +math.inf
        for m in moves:
            new_horses, new_board = apply_move(snap['horses'], snap['board'], m['horseId'], m['move'])
            child_snap = {
                'horses': new_horses,
                'board': new_board,
                'currentPlayer': 3 - snap['currentPlayer'],
                'lastMoveToCenter': m['move']['col'] == CENTER[0] and m['move']['row'] == CENTER[1]
            }
            val = minimax(child_snap, depth - 1, alpha, beta, True, ai_player, deadline)
            best = min(best, val)
            beta = min(beta, val)
            if beta <= alpha: break
        return best


# ===== Root Search =====

def get_best_move(horses, board, player, depth=5, time_limit=2.0, randomise_equal=True):
    """
    Mirrors getAIMove() in ai.js with iterative deepening.
    """
    snap = {
        'horses': horses,
        'board': board,
        'currentPlayer': player,
        'lastMoveToCenter': False
    }
    moves = generate_ordered_moves(snap)

    if not moves:
        return None

    # Immediate win check
    first_move = moves[0]
    if (first_move['move']['col'], first_move['move']['row']) == CENTER:
        return first_move['horseId'], first_move['move']

    time_budget = time_limit
    deadline = time.time() + time_budget
    best_val = -math.inf
    best_move = moves[0]

    for d in range(3, depth + 1):
        if time.time() > deadline: break

        depth_best_val = -math.inf
        depth_best_move = moves[0]
        completed = True

        for candidate in moves:
            if time.time() > deadline:
                completed = False
                break
            
            new_horses, new_board = apply_move(snap['horses'], snap['board'], candidate['horseId'], candidate['move'])
            child_snap = {
                'horses': new_horses,
                'board': new_board,
                'currentPlayer': 3 - snap['currentPlayer'],
                'lastMoveToCenter': candidate['move']['col'] == CENTER[0] and candidate['move']['row'] == CENTER[1]
            }

            val = minimax(child_snap, d - 1, -math.inf, math.inf, False, player, deadline)
            
            if val > depth_best_val:
                depth_best_val = val
                depth_best_move = candidate

        if completed or depth_best_val > best_val:
            best_val = depth_best_val
            best_move = depth_best_move

    return best_move['horseId'], best_move['move']


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

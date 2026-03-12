"""
game.py — Python port of the Squirrel Oasis board/move logic.
Mirrors game.js exactly so the Python minimax matches the JS AI.

Board representation:
  board: dict mapping (col, row) -> owner (1 or 2)
  horses: list of dicts  {id, owner, col, row}
  currentPlayer: 1 or 2

All coordinates are 1-indexed, range [1, 11].
Center is (6, 6).
"""

# ===== Constants =====

BOARD_SIZE = 11
CENTER = (6, 6)

KNIGHT_OFFSETS = [
    (+2, +1), (+2, -1), (-2, +1), (-2, -1),
    (+1, +2), (+1, -2), (-1, +2), (-1, -2),
]

# Starting positions per corner key
CORNER_CONFIGS = {
    'TL': [(1,1),(2,1),(3,1),(1,2),(1,3)],
    'TR': [(11,1),(10,1),(9,1),(11,2),(11,3)],
    'BL': [(1,11),(2,11),(3,11),(1,10),(1,9)],
    'BR': [(11,11),(10,11),(9,11),(11,10),(11,9)],
}
PLAYER_CORNERS = {
    1: ['TL', 'BR'],
    2: ['TR', 'BL'],
}

ZONE_DESERT = 'desert'
ZONE_FOREST = 'forest'
ZONE_OASIS  = 'oasis'


# ===== Zone Helpers =====

def get_zone(col, row):
    dist = abs(col - CENTER[0]) + abs(row - CENTER[1])
    if dist == 0:
        return ZONE_OASIS
    if dist <= 2:
        return ZONE_FOREST
    return ZONE_DESERT


def is_on_board(col, row):
    return 1 <= col <= BOARD_SIZE and 1 <= row <= BOARD_SIZE


# ===== Initial Setup =====

def make_initial_horses():
    """Return horses list and board dict for a fresh game."""
    horses = []
    board  = {}
    for player in [1, 2]:
        idx = 0
        for corner_key in PLAYER_CORNERS[player]:
            for (col, row) in CORNER_CONFIGS[corner_key]:
                h = {'id': f'p{player}_h{idx}', 'owner': player, 'col': col, 'row': row}
                horses.append(h)
                board[(col, row)] = h
                idx += 1
    return horses, board


def horses_to_board(horses):
    """Rebuild board dict from horses list."""
    return {(h['col'], h['row']): h for h in horses}


# ===== Move Generation =====

def slide_ray(horse, dc, dr, board):
    """Slide in one direction; return endpoint dict or None."""
    c, r = horse['col'] + dc, horse['row'] + dr
    last_c, last_r = None, None
    while is_on_board(c, r) and (c, r) not in board:
        last_c, last_r = c, r
        c += dc
        r += dr
    if last_c is None:
        return None
    return {'col': last_c, 'row': last_r, 'move_type': 'slide'}


def slide_moves(horse, board):
    moves = []
    for dc, dr in [(0,-1),(0,1),(-1,0),(1,0)]:
        m = slide_ray(horse, dc, dr, board)
        if m:
            moves.append(m)
    return moves


def knight_moves(horse, board):
    moves = []
    for dc, dr in KNIGHT_OFFSETS:
        c, r = horse['col'] + dc, horse['row'] + dr
        if not is_on_board(c, r):
            continue
        if (c, r) in board:
            continue
        if get_zone(c, r) != ZONE_DESERT:
            continue
        moves.append({'col': c, 'row': r, 'move_type': 'knight'})
    return moves


def valid_moves(horse, board):
    return slide_moves(horse, board) + knight_moves(horse, board)


# ===== State Transitions =====

def apply_move(horses, board, horse_id, move):
    """
    Return a new (horses, board, next_player) triple.
    Does NOT mutate the inputs.
    """
    new_horses = [
        dict(h, col=move['col'], row=move['row']) if h['id'] == horse_id else dict(h)
        for h in horses
    ]
    new_board = horses_to_board(new_horses)
    return new_horses, new_board


def is_won(board):
    """True if any piece is at the center."""
    return CENTER in board


# ===== Board Symmetry (for data augmentation) =====

def rotate_180(col, row):
    """180° rotation: (col,row) → (12-col, 12-row). Center (6,6) maps to (6,6)."""
    return 12 - col, 12 - row


def augment_horses_180(horses):
    """Return a new horses list with all positions rotated 180°. Player assignments unchanged."""
    return [dict(h, col=12 - h['col'], row=12 - h['row']) for h in horses]


# ===== Feature Extraction =====

def state_to_features(horses, current_player):
    """
    Encode a board position as a 21-element list of floats in [0, 1].

    Layout:
      [0..9]  P1 pieces sorted by (col, row): 5 × (col-1)/10, (row-1)/10
      [10..19] P2 pieces sorted by (col, row): 5 × (col-1)/10, (row-1)/10
      [20]    current player: 0.0 if P1, 1.0 if P2
    """
    p1 = sorted([h for h in horses if h['owner'] == 1], key=lambda h: (h['col'], h['row']))
    p2 = sorted([h for h in horses if h['owner'] == 2], key=lambda h: (h['col'], h['row']))
    feats = []
    for h in p1 + p2:
        feats.append((h['col'] - 1) / 10.0)
        feats.append((h['row'] - 1) / 10.0)
    feats.append(0.0 if current_player == 1 else 1.0)
    return feats  # length 21

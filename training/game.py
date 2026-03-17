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


# ===== Feature Extraction Helpers =====

def count_winning_threats(horse_list, board, player):
    """Count horses with on-axis + clear path to center + matching backstop occupied by player."""
    count = 0
    for h in horse_list:
        col, row = h['col'], h['row']
        if col == CENTER[0] and row == CENTER[1]:
            continue  # already at center
        if col != CENTER[0] and row != CENTER[1]:
            continue  # not on axis
        dc = 0 if col == CENTER[0] else (1 if col < CENTER[0] else -1)
        dr = 0 if row == CENTER[1] else (1 if row < CENTER[1] else -1)
        c, r = col + dc, row + dr
        clear = True
        while (c, r) != CENTER:
            if (c, r) in board:
                clear = False
                break
            c += dc
            r += dr
        if not clear:
            continue
        # Determine required backstop cell for this approach direction
        if col == CENTER[0]:
            bs = (6, 7) if row < CENTER[1] else (6, 5)
        else:
            bs = (7, 6) if col < CENTER[0] else (5, 6)
        occupant = board.get(bs)
        if occupant is not None and occupant['owner'] == player:
            count += 1
    return count


def is_in_home(h, player):
    """True if horse is in its starting corner region."""
    col, row = h['col'], h['row']
    if player == 1:
        return (col <= 3 and row <= 3) or (col >= 9 and row >= 9)
    else:
        return (col >= 9 and row <= 3) or (col <= 3 and row >= 9)


def count_pieces_blocking_opp(my_horses, opp_horses):
    """Count my axis pieces that block an opponent piece further back on the same axis approach."""
    blocking = 0
    for my_h in my_horses:
        my_col, my_row = my_h['col'], my_h['row']
        if my_col != CENTER[0] and my_row != CENTER[1]:
            continue  # not on axis
        if my_col == CENTER[0] and my_row == CENTER[1]:
            continue  # skip center itself
        is_blocking = False
        for opp_h in opp_horses:
            opp_col, opp_row = opp_h['col'], opp_h['row']
            if my_col == CENTER[0] and opp_col == CENTER[0]:
                if opp_row == CENTER[1]:
                    continue
                # Same column axis: my piece blocks opp if same side and closer to center
                if (my_row < CENTER[1]) == (opp_row < CENTER[1]):
                    if abs(my_row - CENTER[1]) < abs(opp_row - CENTER[1]):
                        is_blocking = True
                        break
            elif my_row == CENTER[1] and opp_row == CENTER[1]:
                if opp_col == CENTER[0]:
                    continue
                # Same row axis: my piece blocks opp if same side and closer to center
                if (my_col < CENTER[0]) == (opp_col < CENTER[0]):
                    if abs(my_col - CENTER[0]) < abs(opp_col - CENTER[0]):
                        is_blocking = True
                        break
        if is_blocking:
            blocking += 1
    return blocking


# ===== Feature Extraction =====

def state_to_features(horses, current_player, board=None):
    """
    Encode a board position as a 110-element list of floats.

    Per horse × 20 horses = 100 features (P1 first, then P2, each sorted by col/row):
      [base+0] col_norm        = (col - 1) / 10
      [base+1] row_norm        = (row - 1) / 10
      [base+2] dist_to_center  = (|col-6| + |row-6|) / 10
      [base+3] on_axis         = 1.0 if col==6 or row==6, else 0.0
      [base+4] path_threat     = graded 0.0–1.0: max(0, (5-blocks)/5) if on_axis, else 0.0
                                  1.0=clear path, 0.8=1 blocker, 0.6=2 blockers, 0.0=not on axis

    Global features [100..109]:
      [100] backstop (6,5) — +1.0=current player, -1.0=opponent, 0.0=empty
      [101] backstop (6,7) — same
      [102] backstop (5,6) — same
      [103] backstop (7,6) — same
      [104] player indicator — 0.0 if P1, 1.0 if P2
      [105] my_winning_threats  / 10  — horses with on_axis + clear path + backstop set by me
      [106] opp_winning_threats / 10  — same for opponent
      [107] my_horses_at_home   / 10  — my pieces still in starting corner regions
      [108] opp_horses_at_home  / 10  — opponent pieces still in starting corner regions
      [109] my_pieces_blocking_opp / 10 — my axis pieces blocking an opp piece further back
    """
    if board is None:
        board = horses_to_board(horses)

    p1 = sorted([h for h in horses if h['owner'] == 1], key=lambda h: (h['col'], h['row']))
    p2 = sorted([h for h in horses if h['owner'] == 2], key=lambda h: (h['col'], h['row']))

    feats = []
    for h in p1 + p2:
        col, row = h['col'], h['row']

        feats.append((col - 1) / 10.0)
        feats.append((row - 1) / 10.0)
        feats.append((abs(col - CENTER[0]) + abs(row - CENTER[1])) / 10.0)

        on_axis = 1.0 if (col == CENTER[0] or row == CENTER[1]) else 0.0
        feats.append(on_axis)

        path_threat = 0.0
        if on_axis:
            if col == CENTER[0] and row == CENTER[1]:
                path_threat = 1.0  # already at center (terminal state)
            else:
                dc = 0 if col == CENTER[0] else (1 if col < CENTER[0] else -1)
                dr = 0 if row == CENTER[1] else (1 if row < CENTER[1] else -1)
                c, r = col + dc, row + dr
                blocks = 0
                while (c, r) != CENTER:
                    if (c, r) in board:
                        blocks += 1
                    c += dc
                    r += dr
                path_threat = max(0.0, (5 - blocks) / 5.0)
        feats.append(path_threat)

    # Backstop cells — sign relative to current player
    for bs_col, bs_row in [(6, 5), (6, 7), (5, 6), (7, 6)]:
        occupant = board.get((bs_col, bs_row))
        if occupant is None:
            feats.append(0.0)
        elif occupant['owner'] == current_player:
            feats.append(1.0)
        else:
            feats.append(-1.0)

    feats.append(0.0 if current_player == 1 else 1.0)

    # New global features [105..109]
    opp_player = 3 - current_player
    my_horses  = [h for h in horses if h['owner'] == current_player]
    opp_horses = [h for h in horses if h['owner'] == opp_player]

    feats.append(count_winning_threats(my_horses,  board, current_player) / 10.0)   # [105]
    feats.append(count_winning_threats(opp_horses, board, opp_player)     / 10.0)   # [106]
    feats.append(sum(1 for h in my_horses  if is_in_home(h, current_player)) / 10.0) # [107]
    feats.append(sum(1 for h in opp_horses if is_in_home(h, opp_player))     / 10.0) # [108]
    feats.append(count_pieces_blocking_opp(my_horses, opp_horses)             / 10.0) # [109]

    return feats  # length 110

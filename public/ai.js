'use strict';

// ===== Neural Network Evaluation =====
//
// When public/model/weights.json exists (produced by the training pipeline),
// nnEvaluate() is used as the leaf evaluation function inside minimax instead
// of the handcrafted evaluate().
//
// The model is a small MLP: Input(21) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(1, Tanh)
// Output in [-1, +1]: positive = good for the player to move, negative = bad.
//
// If weights.json is absent, nnEvaluate() transparently falls back to evaluate().

let NNWeights = null;

async function loadNNModel() {
  try {
    const resp = await fetch('model/weights.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    NNWeights = await resp.json();
    console.log('[AI] Neural net weights loaded — using NN evaluation.');
  } catch (e) {
    console.info('[AI] No weights.json found — using heuristic evaluation. ' +
                 '(Run the training pipeline to generate weights.)');
  }
}

// Feature extraction — mirrors state_to_features() in training/game.py
function stateToFeatures(snap) {
  const p1 = snap.horses.filter(h => h.owner === 1)
    .sort((a, b) => a.col - b.col || a.row - b.row);
  const p2 = snap.horses.filter(h => h.owner === 2)
    .sort((a, b) => a.col - b.col || a.row - b.row);
  const feats = [];
  for (const h of [...p1, ...p2]) {
    feats.push((h.col - 1) / 10);
    feats.push((h.row - 1) / 10);
  }
  feats.push(snap.currentPlayer === 1 ? 0.0 : 1.0);
  return feats;  // length 41 (10 pieces × 2 coords × 2 players + 1 player indicator)
}

// One fully-connected layer: output[i] = ReLU(b[i] + sum_j W[i][j] * x[j])
function nnLayer(x, W, b, activation) {
  return b.map((bi, i) => {
    const z = bi + W[i].reduce((s, w, j) => s + w * x[j], 0);
    return activation === 'relu' ? Math.max(0, z) : Math.tanh(z);
  });
}

// Replace handcrafted evaluate() with neural net when weights are loaded.
// Falls back to evaluate() if model is not yet loaded.
function nnEvaluate(snap) {
  if (!NNWeights) return evaluate(snap);

  const x  = stateToFeatures(snap);
  const h1 = nnLayer(x,  NNWeights.W1, NNWeights.b1, 'relu');   // 41 → 128
  const h2 = nnLayer(h1, NNWeights.W2, NNWeights.b2, 'relu');   // 128 → 64
  const out = nnLayer(h2, NNWeights.W3, NNWeights.b3, 'tanh');  // 64 → 1

  // out[0] ∈ [-1, 1]: positive = good for the player to move.
  // Minimax always maximises for AIState.playerNumber, so orient the sign:
  const oriented = snap.currentPlayer === AIState.playerNumber ? out[0] : -out[0];
  return oriented * 10000;  // scale to match heuristic score range
}

// ===== AI State + Entry Points (external API) =====

const AIState = {
  enabled: false,
  playerNumber: 2,   // AI always plays as Black (Player 2)
  thinking: false,
};

function startVsComputer() {
  AIState.enabled = true;
  AIState.thinking = false;
  startGame();
}

// Called after each move; no-ops if it's not AI's turn
function scheduleAIMove() {
  if (!AIState.enabled) return;
  if (GameState.currentPlayer !== AIState.playerNumber) return;
  if (GameState.phase !== 'playing') return;

  AIState.thinking = true;
  render();  // show "thinking" status

  const delay = 400 + Math.random() * 300;
  setTimeout(() => {
    if (GameState.phase !== 'playing') return;
    AIState.thinking = false;
    const result = getAIMove();
    if (result) executeMove(result.horse, result.move);
  }, delay);
}

// ===== Constants =====

// The 4 backstop cells and the approach direction each enables.
// A piece can ONLY stop at center if the cell immediately past center
// (in the slide direction) is occupied — that's the "backstop".
//   backstop: the cell that must be occupied
//   filter:   which attacking pieces can exploit this backstop
//             (piece must be in same row/col as center, on the correct side)
const APPROACH_CONFIGS = [
  // Attacker above center (col=6, row<6), slides DOWN, needs backstop at (6,7)
  { bs: { col: 6, row: 7 }, filter: h => h.col === CENTER.col && h.row < CENTER.row },
  // Attacker below center (col=6, row>6), slides UP, needs backstop at (6,5)
  { bs: { col: 6, row: 5 }, filter: h => h.col === CENTER.col && h.row > CENTER.row },
  // Attacker left of center (row=6, col<6), slides RIGHT, needs backstop at (7,6)
  { bs: { col: 7, row: 6 }, filter: h => h.row === CENTER.row && h.col < CENTER.col },
  // Attacker right of center (row=6, col>6), slides LEFT, needs backstop at (5,6)
  { bs: { col: 5, row: 6 }, filter: h => h.row === CENTER.row && h.col > CENTER.col },
];

// ===== Pure State Helpers =====

// Lightweight snapshot for minimax — never touches global GameState during search.
function snapshotFromGameState() {
  const horses = GameState.horses.map(h => ({ id: h.id, owner: h.owner, col: h.col, row: h.row }));
  const board = {};
  for (const h of horses) board[boardKey(h.col, h.row)] = h;
  return { horses, board, currentPlayer: GameState.currentPlayer, lastMoveToCenter: false };
}

// Pure state transition. Returns a brand-new snapshot (immutable style).
function applyMove(snap, horseId, move) {
  const horses = snap.horses.map(h =>
    h.id === horseId ? { ...h, col: move.col, row: move.row } : h
  );
  const board = {};
  for (const h of horses) board[boardKey(h.col, h.row)] = h;
  return {
    horses,
    board,
    currentPlayer: 3 - snap.currentPlayer,
    lastMoveToCenter: move.col === CENTER.col && move.row === CENTER.row,
  };
}

// ===== Pure Move Generators =====

// Slide in one direction; returns the endpoint cell or null if the piece can't move.
function pureSlideRay(horse, dc, dr, board) {
  let c = horse.col + dc, r = horse.row + dr;
  while (c >= 1 && c <= BOARD_SIZE && r >= 1 && r <= BOARD_SIZE && !board[boardKey(c, r)]) {
    c += dc; r += dr;
  }
  // Step back to last valid cell
  c -= dc; r -= dr;
  if (c === horse.col && r === horse.row) return null; // couldn't move at all
  return { col: c, row: r, moveType: 'slide' };
}

function pureSlideMoves(horse, board) {
  const moves = [];
  for (const [dc, dr] of [[0,-1],[0,1],[-1,0],[1,0]]) {
    const m = pureSlideRay(horse, dc, dr, board);
    if (m) moves.push(m);
  }
  return moves;
}

// Knight destinations must land in DESERT zone (same rule as game.js line 107).
function pureKnightMoves(horse, board) {
  const moves = [];
  for (const [dc, dr] of KNIGHT_OFFSETS) {
    const c = horse.col + dc, r = horse.row + dr;
    if (c < 1 || c > BOARD_SIZE || r < 1 || r > BOARD_SIZE) continue;
    if (board[boardKey(c, r)]) continue;
    if (getZone(c, r) !== ZONE.DESERT) continue;
    moves.push({ col: c, row: r, moveType: 'knight' });
  }
  return moves;
}

function pureValidMoves(horse, board) {
  return [...pureSlideMoves(horse, board), ...pureKnightMoves(horse, board)];
}

// ===== Threat Analysis =====

// True if every cell strictly between fromHorse and CENTER (along the same row or col) is empty.
function pathClear(fromHorse, board) {
  const dc = Math.sign(CENTER.col - fromHorse.col);
  const dr = Math.sign(CENTER.row - fromHorse.row);
  if (dc !== 0 && dr !== 0) return false; // not aligned
  let c = fromHorse.col + dc, r = fromHorse.row + dr;
  while (!(c === CENTER.col && r === CENTER.row)) {
    if (board[boardKey(c, r)]) return false;
    c += dc; r += dr;
  }
  return true;
}

// For each of the 4 approach axes, compute a score contribution based on
// who occupies the backstop cell and who has an aligned attacker with a clear path.
function scoreBackstopControl(snap) {
  const ai  = AIState.playerNumber;
  const opp = 3 - ai;
  let score = 0;

  for (const { bs, filter } of APPROACH_CONFIGS) {
    const occupant = snap.board[boardKey(bs.col, bs.row)];
    if (!occupant) continue;

    // Find any aligned attacker with a clear path on this axis
    for (const h of snap.horses) {
      if (!filter(h)) continue;
      if (!pathClear(h, snap.board)) continue;

      // A live attack exists. Score based on who attacks and who backstops.
      if (h.owner === ai && occupant.owner === ai) {
        score += 1500;  // AI has set up a winning position
      } else if (h.owner === opp && occupant.owner === opp) {
        score -= 2000;  // Opponent is about to win
      } else if (h.owner === opp && occupant.owner === ai) {
        score -= 800;   // AI piece is opponent's backstop — the critical mistake
      } else if (h.owner === ai && occupant.owner === opp) {
        score += 500;   // Opponent piece acts as AI's backstop (opportunistic)
      }
    }
  }
  return score;
}

// ===== Evaluation =====

function evaluate(snap) {
  const ai  = AIState.playerNumber;
  const opp = 3 - ai;
  let score = 0;

  // Core: backstop-aware threat scoring
  score += scoreBackstopControl(snap);

  // Near-win bonus: aligned + clear path, but backstop not yet in place
  for (const { bs, filter } of APPROACH_CONFIGS) {
    if (snap.board[boardKey(bs.col, bs.row)]) continue; // already handled above
    for (const h of snap.horses) {
      if (!filter(h)) continue;
      if (!pathClear(h, snap.board)) continue;
      // Attacker is aligned and path is clear but no backstop yet
      score += h.owner === ai ? 300 : -400;
    }
  }

  // General positioning — closeness to center, but penalise forest clustering
  for (const h of snap.horses) {
    const dist = Math.abs(h.col - CENTER.col) + Math.abs(h.row - CENTER.row);
    const inForest = dist > 0 && dist <= 2;
    let pos = (10 - dist) * 8;
    if (inForest) pos *= 0.3; // being crammed in forest without a backstop is low value
    if (h.owner === ai) score += pos;
    else                score -= pos;
  }

  return score;
}

// ===== Move Ordering =====

// Returns [{horseId, move}] for the current player, sorted for good alpha-beta cutoffs.
function generateOrderedMoves(snap) {
  const ai  = AIState.playerNumber;
  const player = snap.currentPlayer;
  const moves = [];

  for (const h of snap.horses) {
    if (h.owner !== player) continue;
    for (const move of pureValidMoves(h, snap.board)) {
      moves.push({ horseId: h.id, move });
    }
  }

  // Priority scoring for ordering (not the same as evaluate — just for ordering)
  moves.forEach(m => {
    const { move } = m;
    let pri = 0;
    // 1. Immediate win
    if (move.col === CENTER.col && move.row === CENTER.row) { pri = 1e9; }
    else {
      // 2. Creating a winning backstop for an aligned friendly attacker
      for (const { bs, filter } of APPROACH_CONFIGS) {
        if (move.col === bs.col && move.row === bs.row) {
          // Does a friendly attacker exist with a clear path?
          for (const h of snap.horses) {
            if (h.owner !== player || !filter(h)) continue;
            if (pathClear(h, snap.board)) { pri = Math.max(pri, 8000); }
          }
        }
      }
      // 3. Blocking opponent's aligned attacker
      const opp = 3 - player;
      for (const { filter } of APPROACH_CONFIGS) {
        for (const h of snap.horses) {
          if (h.owner !== opp || !filter(h)) continue;
          if (!pathClear(h, snap.board)) continue;
          // Blocking moves: placing on the path between attacker and center
          const dc = Math.sign(CENTER.col - h.col);
          const dr = Math.sign(CENTER.row - h.row);
          let c = h.col + dc, r = h.row + dr;
          while (!(c === CENTER.col && r === CENTER.row)) {
            if (move.col === c && move.row === r) pri = Math.max(pri, 5000);
            c += dc; r += dr;
          }
        }
      }
      // 4. Fallback: proximity to center
      if (pri === 0) {
        const dist = Math.abs(move.col - CENTER.col) + Math.abs(move.row - CENTER.row);
        pri = 10 - dist;
      }
    }
    m.priority = pri;
  });

  moves.sort((a, b) => b.priority - a.priority);
  return moves;
}

// ===== Minimax =====

function minimax(snap, depth, alpha, beta, maximizing, startTime) {
  // Time guard — avoids going over budget; returns bound rather than ±Infinity
  // so an incomplete subtree doesn't poison the parent's choice.
  if (Date.now() - startTime > 750) {
    return maximizing ? alpha : beta;
  }

  // Terminal: the player who just moved landed on center
  if (snap.lastMoveToCenter) {
    const winner = 3 - snap.currentPlayer; // currentPlayer already flipped in applyMove
    return winner === AIState.playerNumber ? +10000 : -10000;
  }

  if (depth === 0) return nnEvaluate(snap);

  const moves = generateOrderedMoves(snap);
  if (!moves.length) return nnEvaluate(snap);

  if (maximizing) {
    let best = -Infinity;
    for (const { horseId, move } of moves) {
      const child = applyMove(snap, horseId, move);
      const val = minimax(child, depth - 1, alpha, beta, false, startTime);
      if (val > best) best = val;
      if (val > alpha) alpha = val;
      if (alpha >= beta) break;
    }
    return best;
  } else {
    let best = +Infinity;
    for (const { horseId, move } of moves) {
      const child = applyMove(snap, horseId, move);
      const val = minimax(child, depth - 1, alpha, beta, true, startTime);
      if (val < best) best = val;
      if (val < beta) beta = val;
      if (beta <= alpha) break;
    }
    return best;
  }
}

// ===== Root Search =====

// Maps a snapshot horseId back to the actual GameState horse reference that
// executeMove() needs (it mutates the horse object in place).
function resolveToGameObjects({ horseId, move }) {
  const horse = GameState.horses.find(h => h.id === horseId);
  return horse ? { horse, move } : null;
}

function getAIMove() {
  const snap = snapshotFromGameState();
  const startTime = Date.now();

  const moves = generateOrderedMoves(snap);
  if (!moves.length) return null;

  // Take an immediate win without searching
  const winMove = moves[0];
  if (winMove.move.col === CENTER.col && winMove.move.row === CENTER.row) {
    return resolveToGameObjects(winMove);
  }

  let bestVal = -Infinity;
  let bestMove = moves[0];

  for (const candidate of moves) {
    const child = applyMove(snap, candidate.horseId, candidate.move);
    const val = minimax(child, 3, -Infinity, +Infinity, false, startTime);
    if (val > bestVal) {
      bestVal = val;
      bestMove = candidate;
    }
  }

  return resolveToGameObjects(bestMove);
}

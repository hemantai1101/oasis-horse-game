'use strict';

// ===== Neural Network Evaluation =====
//
// When public/model/weights.json exists (produced by the training pipeline),
// nnEvaluate() is used as the leaf evaluation function inside minimax instead
// of the handcrafted evaluate().
//
// The model is a small MLP: Input(41) -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(1, Tanh)
// Output in [-1, +1]: positive = good for the player to move, negative = bad.
//
// If weights.json is absent, nnEvaluate() transparently falls back to evaluate().

let NNWeights = null;

// Set to true to force heuristic evaluation even when weights are loaded.
// Useful for testing the new heuristic without retraining the NN.
const FORCE_HEURISTIC = false;

async function loadNNModel() {
  try {
    const resp = await fetch('model/weights.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const w = await resp.json();
    if (!w.bn1_mean) {
      console.warn('[AI] weights.json is missing BatchNorm params — re-run export_weights.py. ' +
        'Falling back to heuristic evaluation.');
    } else {
      NNWeights = w;
    }
    if (!NNWeights) {
      // weights loaded but incomplete
    } else if (FORCE_HEURISTIC) {
      console.log('[AI] Neural net weights loaded.');
      console.log('[AI] FORCE_HEURISTIC=true: using heuristic evaluation (NN ignored).');
    } else {
      console.log('[AI] Neural net weights loaded.');
      console.log('[AI] Using NN evaluation.');
    }
  } catch (e) {
    console.info('[AI] No weights.json found: using heuristic evaluation. ' +
      '(Run the training pipeline to generate weights.)');
  }
}

// Feature extraction -- mirrors state_to_features() in training/game.py
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
  return feats;  // length 41 (10 pieces x 2 coords x 2 players + 1 player indicator)
}

// One fully-connected layer: output[i] = activation(b[i] + sum_j W[i][j] * x[j])
function nnLayer(x, W, b, activation) {
  return b.map((bi, i) => {
    const z = bi + W[i].reduce((s, w, j) => s + w * x[j], 0);
    return activation === 'relu' ? Math.max(0, z) : Math.tanh(z);
  });
}

// BatchNorm inference: y = (x - mean) / sqrt(var + eps) * gamma + beta
function nnBatchNorm(x, mean, variance, gamma, beta, eps) {
  return x.map((xi, i) => {
    const xhat = (xi - mean[i]) / Math.sqrt(variance[i] + eps);
    return gamma[i] * xhat + beta[i];
  });
}

// Replace handcrafted evaluate() with neural net when weights are loaded.
// Falls back to evaluate() if model is not yet loaded or FORCE_HEURISTIC is set.
function nnEvaluate(snap) {
  if (!NNWeights || FORCE_HEURISTIC) return evaluate(snap);

  const W = NNWeights;
  const x   = stateToFeatures(snap);
  const l1  = nnLayer(x,  W.W1, W.b1, 'relu');                                        // 41 -> 128
  const bn1 = nnBatchNorm(l1, W.bn1_mean, W.bn1_var, W.bn1_gamma, W.bn1_beta, W.bn1_eps);
  const l2  = nnLayer(bn1, W.W2, W.b2, 'relu');                                        // 128 -> 64
  const bn2 = nnBatchNorm(l2, W.bn2_mean, W.bn2_var, W.bn2_gamma, W.bn2_beta, W.bn2_eps);
  const out = nnLayer(bn2, W.W3, W.b3, 'tanh');                                        // 64 -> 1

  // out[0] in [-1, 1]: positive = good for the player to move.
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

function snapshotFromGameState() {
  const horses = GameState.horses.map(h => ({ id: h.id, owner: h.owner, col: h.col, row: h.row }));
  const board = {};
  for (const h of horses) board[boardKey(h.col, h.row)] = h;
  return { horses, board, currentPlayer: GameState.currentPlayer, lastMoveToCenter: false };
}

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

function pureSlideRay(horse, dc, dr, board) {
  let c = horse.col + dc, r = horse.row + dr;
  while (c >= 1 && c <= BOARD_SIZE && r >= 1 && r <= BOARD_SIZE && !board[boardKey(c, r)]) {
    c += dc; r += dr;
  }
  c -= dc; r -= dr;
  if (c === horse.col && r === horse.row) return null;
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

// Count how many moves a piece has (pure, no side effects)
function countMobility(horse, board) {
  return pureValidMoves(horse, board).length;
}

// ===== EVALUATION (complete rewrite) =====

function evaluate(snap) {
  const ai  = AIState.playerNumber;
  const opp = 3 - ai;
  let score = 0;

  const aiHorses  = snap.horses.filter(h => h.owner === ai);
  const oppHorses = snap.horses.filter(h => h.owner === opp);

  // ---------------------------------------------------------------
  // 1. TERMINAL THREAT SCORING (highest priority)
  //    A "ready attack" = attacker aligned on axis with clear path +
  //    backstop cell occupied by a friendly piece.
  //    A "near attack" = attacker aligned but backstop empty.
  // ---------------------------------------------------------------

  for (const { bs, filter } of APPROACH_CONFIGS) {
    const occupant = snap.board[boardKey(bs.col, bs.row)];

    // --- Ready attacks (backstop occupied) ---
    if (occupant) {
      for (const h of snap.horses) {
        if (!filter(h)) continue;
        if (!pathClear(h, snap.board)) continue;

        if (h.owner === ai && occupant.owner === ai) {
          score += 3000;  // AI has a winning setup
        } else if (h.owner === opp && occupant.owner === opp) {
          score -= 4000;  // Opponent about to win -- massive penalty
        } else if (h.owner === opp && occupant.owner === ai) {
          score -= 1200;  // AI piece is being used as opponent's backstop
        } else if (h.owner === ai && occupant.owner === opp) {
          score += 800;   // Opponent piece acts as AI's backstop
        }
      }
    } else {
      // --- Near attacks (no backstop yet) ---
      for (const h of snap.horses) {
        if (!filter(h)) continue;
        if (!pathClear(h, snap.board)) continue;
        score += h.owner === ai ? 400 : -600;
      }
    }
  }

  // ---------------------------------------------------------------
  // 2. PIECE DEVELOPMENT & POSITIONING
  //    Reward advancing pieces out of corners towards the center axis.
  //    Heavier weights than before so the AI actually moves pieces.
  // ---------------------------------------------------------------

  for (const h of snap.horses) {
    const dist   = Math.abs(h.col - CENTER.col) + Math.abs(h.row - CENTER.row);
    const onAxis = h.col === CENTER.col || h.row === CENTER.row;
    const sign   = h.owner === ai ? 1 : -1;

    // Base proximity score: being closer to center is good
    // Max dist on 11x11 = 10, so (10-dist)*15 ranges from 0 to 150
    score += sign * (10 - dist) * 15;

    // Axis alignment bonus: being on col=6 or row=6 is strategically critical
    if (onAxis) {
      score += sign * 100;
      // Extra bonus if close to center on axis (within 3 steps)
      if (dist <= 3) {
        score += sign * 80;
      }
    }

    // Corner penalty: pieces stuck in starting corners are useless
    const inCorner = (
      (h.col <= 3 && h.row <= 3) ||
      (h.col <= 3 && h.row >= 9) ||
      (h.col >= 9 && h.row <= 3) ||
      (h.col >= 9 && h.row >= 9)
    );
    if (inCorner) {
      score -= sign * 60;
    }

    // Backstop cell occupation bonus (even without aligned attacker)
    const key = boardKey(h.col, h.row);
    if (key === '6,5' || key === '6,7' || key === '5,6' || key === '7,6') {
      score += sign * 120;
    }
  }

  // ---------------------------------------------------------------
  // 3. MOBILITY
  //    Having more available moves = more options = better position.
  //    This also naturally encourages developing multiple pieces.
  // ---------------------------------------------------------------

  let aiMobility = 0, oppMobility = 0;
  for (const h of aiHorses)  aiMobility  += countMobility(h, snap.board);
  for (const h of oppHorses) oppMobility += countMobility(h, snap.board);
  score += (aiMobility - oppMobility) * 8;

  // ---------------------------------------------------------------
  // 4. PIECE SPREAD / DEVELOPMENT DIVERSITY
  //    Penalise having too many pieces clustered in one area.
  //    Reward having pieces spread across the board.
  // ---------------------------------------------------------------

  // Count how many AI pieces have left their starting corners
  let aiDeveloped = 0;
  for (const h of aiHorses) {
    const inStartCorner = (
      (h.col >= 9 && h.row <= 3) ||  // TR corner (P2 start)
      (h.col <= 3 && h.row >= 9)     // BL corner (P2 start)
    );
    if (!inStartCorner) aiDeveloped++;
  }
  // Reward developing pieces (each piece that leaves the corner)
  score += aiDeveloped * 25;

  let oppDeveloped = 0;
  for (const h of oppHorses) {
    const inStartCorner = (
      (h.col <= 3 && h.row <= 3) ||  // TL corner (P1 start)
      (h.col >= 9 && h.row >= 9)     // BR corner (P1 start)
    );
    if (!inStartCorner) oppDeveloped++;
  }
  score -= oppDeveloped * 25;

  // ---------------------------------------------------------------
  // 5. AXIS CONTROL
  //    Count pieces on the center row and column (col=6 or row=6).
  //    More pieces on axis = more attack options.
  // ---------------------------------------------------------------

  let aiOnAxis = 0, oppOnAxis = 0;
  for (const h of aiHorses) {
    if (h.col === CENTER.col || h.row === CENTER.row) aiOnAxis++;
  }
  for (const h of oppHorses) {
    if (h.col === CENTER.col || h.row === CENTER.row) oppOnAxis++;
  }
  score += (aiOnAxis - oppOnAxis) * 40;

  // ---------------------------------------------------------------
  // 6. BLOCKING AWARENESS
  //    If opponent has an aligned attacker with clear path, any AI
  //    piece that could move into that path to block gets a bonus.
  // ---------------------------------------------------------------

  for (const { filter } of APPROACH_CONFIGS) {
    for (const h of oppHorses) {
      if (!filter(h)) continue;
      if (!pathClear(h, snap.board)) continue;

      // Opponent has a dangerous alignment. Check if AI can block next turn.
      const dc = Math.sign(CENTER.col - h.col);
      const dr = Math.sign(CENTER.row - h.row);
      let c = h.col + dc, r = h.row + dr;
      const pathCells = [];
      while (!(c === CENTER.col && r === CENTER.row)) {
        pathCells.push({ col: c, row: r });
        c += dc; r += dr;
      }

      // Check if any AI piece can reach a blocking position
      let canBlock = false;
      for (const ah of aiHorses) {
        const moves = pureValidMoves(ah, snap.board);
        for (const m of moves) {
          if (pathCells.some(pc => pc.col === m.col && pc.row === m.row)) {
            canBlock = true;
            break;
          }
        }
        if (canBlock) break;
      }
      // If we can't block, extra penalty
      if (!canBlock) {
        score -= 500;
      }
    }
  }

  return score;
}

// ===== Move Ordering =====

function generateOrderedMoves(snap) {
  const player = snap.currentPlayer;
  const opp = 3 - player;
  const moves = [];

  for (const h of snap.horses) {
    if (h.owner !== player) continue;
    for (const move of pureValidMoves(h, snap.board)) {
      moves.push({ horseId: h.id, move });
    }
  }

  moves.forEach(m => {
    const { move, horseId } = m;
    let pri = 0;

    // 1. Immediate win
    if (move.col === CENTER.col && move.row === CENTER.row) {
      pri = 1e9;
    } else {
      // 2. Creating a winning backstop for an aligned friendly attacker
      for (const { bs, filter } of APPROACH_CONFIGS) {
        if (move.col === bs.col && move.row === bs.row) {
          for (const h of snap.horses) {
            if (h.owner !== player || !filter(h)) continue;
            if (pathClear(h, snap.board)) { pri = Math.max(pri, 8000); }
          }
        }
      }

      // 3. Moving an aligned attacker with clear path (the attack itself)
      for (const { bs, filter } of APPROACH_CONFIGS) {
        const horse = snap.horses.find(h => h.id === horseId);
        if (!horse || !filter(horse)) continue;
        if (!pathClear(horse, snap.board)) continue;
        const occupant = snap.board[boardKey(bs.col, bs.row)];
        if (occupant && occupant.owner === player) {
          pri = Math.max(pri, 7000);
        }
      }

      // 4. Blocking opponent's aligned attacker
      for (const { filter } of APPROACH_CONFIGS) {
        for (const h of snap.horses) {
          if (h.owner !== opp || !filter(h)) continue;
          if (!pathClear(h, snap.board)) continue;
          const dc = Math.sign(CENTER.col - h.col);
          const dr = Math.sign(CENTER.row - h.row);
          let c = h.col + dc, r = h.row + dr;
          while (!(c === CENTER.col && r === CENTER.row)) {
            if (move.col === c && move.row === r) pri = Math.max(pri, 6000);
            c += dc; r += dr;
          }
        }
      }

      // 5. Moving to axis (col=6 or row=6)
      if (pri === 0 && (move.col === CENTER.col || move.row === CENTER.row)) {
        const dist = Math.abs(move.col - CENTER.col) + Math.abs(move.row - CENTER.row);
        pri = Math.max(pri, 100 + (10 - dist) * 10);
      }

      // 6. Fallback: proximity to center
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

// ===== Minimax with Iterative Deepening =====

let nodeCount = 0;

function minimax(snap, depth, alpha, beta, maximizing, deadline) {
  nodeCount++;

  // Time guard
  if (nodeCount % 256 === 0 && Date.now() > deadline) {
    return maximizing ? alpha : beta;
  }

  // Terminal: the player who just moved landed on center
  if (snap.lastMoveToCenter) {
    const winner = 3 - snap.currentPlayer;
    return winner === AIState.playerNumber ? +50000 : -50000;
  }

  if (depth === 0) return nnEvaluate(snap);

  const moves = generateOrderedMoves(snap);
  if (!moves.length) return nnEvaluate(snap);

  if (maximizing) {
    let best = -Infinity;
    for (const { horseId, move } of moves) {
      const child = applyMove(snap, horseId, move);
      const val = minimax(child, depth - 1, alpha, beta, false, deadline);
      if (val > best) best = val;
      if (val > alpha) alpha = val;
      if (alpha >= beta) break;
    }
    return best;
  } else {
    let best = +Infinity;
    for (const { horseId, move } of moves) {
      const child = applyMove(snap, horseId, move);
      const val = minimax(child, depth - 1, alpha, beta, true, deadline);
      if (val < best) best = val;
      if (val < beta) beta = val;
      if (beta <= alpha) break;
    }
    return best;
  }
}

// ===== Root Search =====

function resolveToGameObjects({ horseId, move }) {
  const horse = GameState.horses.find(h => h.id === horseId);
  return horse ? { horse, move } : null;
}

function getAIMove() {
  const snap = snapshotFromGameState();
  const moves = generateOrderedMoves(snap);
  if (!moves.length) return null;

  // Take an immediate win without searching
  const winMove = moves[0];
  if (winMove.move.col === CENTER.col && winMove.move.row === CENTER.row) {
    return resolveToGameObjects(winMove);
  }

  // Iterative deepening: search depth 3, then 4, then 5.
  // Use the best result from the deepest completed search.
  const timeBudget = 12000; // ms -- more generous than before
  const deadline = Date.now() + timeBudget;
  let bestVal = -Infinity;
  let bestMove = moves[0];

  for (let depth = 3; depth <= 6; depth++) {
    if (Date.now() > deadline) break;

    nodeCount = 0;
    let depthBestVal = -Infinity;
    let depthBestMove = moves[0];
    let completed = true;

    for (const candidate of moves) {
      if (Date.now() > deadline) { completed = false; break; }
      const child = applyMove(snap, candidate.horseId, candidate.move);
      const val = minimax(child, depth - 1, -Infinity, +Infinity, false, deadline);
      if (val > depthBestVal) {
        depthBestVal = val;
        depthBestMove = candidate;
      }
    }

    // Only use this depth's result if we completed all moves
    if (completed || depthBestVal > bestVal) {
      bestVal = depthBestVal;
      bestMove = depthBestMove;
    }

    console.log(`[AI] depth=${depth} best=${depthBestVal.toFixed(0)} nodes=${nodeCount} ` +
      `move=${bestMove.horseId} -> (${bestMove.move.col},${bestMove.move.row})`);
  }

  return resolveToGameObjects(bestMove);
}

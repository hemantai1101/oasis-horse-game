'use strict';

// ===== Constants (mirrors game.js) =====

const BOARD_SIZE = 11;
const CENTER = { col: 6, row: 6 };

const ZONE = {
  DESERT: 'desert',
  FOREST: 'forest',
  OASIS:  'oasis',
};

const KNIGHT_OFFSETS = [
  [+2, +1], [+2, -1], [-2, +1], [-2, -1],
  [+1, +2], [+1, -2], [-1, +2], [-1, -2],
];

const APPROACH_CONFIGS = [
  { bs: { col: 6, row: 7 }, filter: h => h.col === CENTER.col && h.row < CENTER.row },
  { bs: { col: 6, row: 5 }, filter: h => h.col === CENTER.col && h.row > CENTER.row },
  { bs: { col: 7, row: 6 }, filter: h => h.row === CENTER.row && h.col < CENTER.col },
  { bs: { col: 5, row: 6 }, filter: h => h.row === CENTER.row && h.col > CENTER.col },
];

// ===== Per-computation state =====

let NNWeights = null;
let aiPlayerNumber = 2;
let FORCE_HEURISTIC = false;

// ===== Helpers =====

function getZone(col, row) {
  const dist = Math.abs(col - CENTER.col) + Math.abs(row - CENTER.row);
  if (dist === 0) return ZONE.OASIS;
  if (dist <= 2)  return ZONE.FOREST;
  return ZONE.DESERT;
}

function boardKey(col, row) { return `${col},${row}`; }

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

// ===== Apply Move (pure, no side effects) =====

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

// ===== Threat Analysis =====

function pathClear(fromHorse, board) {
  const dc = Math.sign(CENTER.col - fromHorse.col);
  const dr = Math.sign(CENTER.row - fromHorse.row);
  if (dc !== 0 && dr !== 0) return false;
  let c = fromHorse.col + dc, r = fromHorse.row + dr;
  while (!(c === CENTER.col && r === CENTER.row)) {
    if (board[boardKey(c, r)]) return false;
    c += dc; r += dr;
  }
  return true;
}

function countMobility(horse, board) {
  return pureValidMoves(horse, board).length;
}

// ===== Neural Network Evaluation =====

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
  return feats;
}

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

function nnEvaluate(snap) {
  if (!NNWeights || FORCE_HEURISTIC) return evaluate(snap);
  const W   = NNWeights;
  const x   = stateToFeatures(snap);
  const l1  = nnLayer(x,  W.W1, W.b1, 'relu');                                        // 41 -> 128
  const bn1 = nnBatchNorm(l1, W.bn1_mean, W.bn1_var, W.bn1_gamma, W.bn1_beta, W.bn1_eps);
  const l2  = nnLayer(bn1, W.W2, W.b2, 'relu');                                        // 128 -> 64
  const bn2 = nnBatchNorm(l2, W.bn2_mean, W.bn2_var, W.bn2_gamma, W.bn2_beta, W.bn2_eps);
  const out = nnLayer(bn2, W.W3, W.b3, 'tanh');                                        // 64 -> 1
  const oriented = snap.currentPlayer === aiPlayerNumber ? out[0] : -out[0];
  return oriented * 10000;
}

// ===== Heuristic Evaluation =====

function evaluate(snap) {
  const ai  = aiPlayerNumber;
  const opp = 3 - ai;
  let score = 0;

  const aiHorses  = snap.horses.filter(h => h.owner === ai);
  const oppHorses = snap.horses.filter(h => h.owner === opp);

  for (const { bs, filter } of APPROACH_CONFIGS) {
    const occupant = snap.board[boardKey(bs.col, bs.row)];
    if (occupant) {
      for (const h of snap.horses) {
        if (!filter(h)) continue;
        if (!pathClear(h, snap.board)) continue;
        if (h.owner === ai && occupant.owner === ai) {
          score += 3000;
        } else if (h.owner === opp && occupant.owner === opp) {
          score -= 4000;
        } else if (h.owner === opp && occupant.owner === ai) {
          score -= 1200;
        } else if (h.owner === ai && occupant.owner === opp) {
          score += 800;
        }
      }
    } else {
      for (const h of snap.horses) {
        if (!filter(h)) continue;
        if (!pathClear(h, snap.board)) continue;
        score += h.owner === ai ? 400 : -600;
      }
    }
  }

  for (const h of snap.horses) {
    const dist   = Math.abs(h.col - CENTER.col) + Math.abs(h.row - CENTER.row);
    const onAxis = h.col === CENTER.col || h.row === CENTER.row;
    const sign   = h.owner === ai ? 1 : -1;
    score += sign * (10 - dist) * 15;
    if (onAxis) {
      score += sign * 100;
      if (dist <= 3) score += sign * 80;
    }
    const inCorner = (
      (h.col <= 3 && h.row <= 3) ||
      (h.col <= 3 && h.row >= 9) ||
      (h.col >= 9 && h.row <= 3) ||
      (h.col >= 9 && h.row >= 9)
    );
    if (inCorner) score -= sign * 60;
    const key = boardKey(h.col, h.row);
    if (key === '6,5' || key === '6,7' || key === '5,6' || key === '7,6') {
      score += sign * 120;
    }
  }

  let aiMobility = 0, oppMobility = 0;
  for (const h of aiHorses)  aiMobility  += countMobility(h, snap.board);
  for (const h of oppHorses) oppMobility += countMobility(h, snap.board);
  score += (aiMobility - oppMobility) * 8;

  let aiDeveloped = 0;
  for (const h of aiHorses) {
    const inStartCorner = (h.col >= 9 && h.row <= 3) || (h.col <= 3 && h.row >= 9);
    if (!inStartCorner) aiDeveloped++;
  }
  score += aiDeveloped * 25;

  let oppDeveloped = 0;
  for (const h of oppHorses) {
    const inStartCorner = (h.col <= 3 && h.row <= 3) || (h.col >= 9 && h.row >= 9);
    if (!inStartCorner) oppDeveloped++;
  }
  score -= oppDeveloped * 25;

  let aiOnAxis = 0, oppOnAxis = 0;
  for (const h of aiHorses)  { if (h.col === CENTER.col || h.row === CENTER.row) aiOnAxis++; }
  for (const h of oppHorses) { if (h.col === CENTER.col || h.row === CENTER.row) oppOnAxis++; }
  score += (aiOnAxis - oppOnAxis) * 40;

  for (const { filter } of APPROACH_CONFIGS) {
    for (const h of oppHorses) {
      if (!filter(h)) continue;
      if (!pathClear(h, snap.board)) continue;
      const dc = Math.sign(CENTER.col - h.col);
      const dr = Math.sign(CENTER.row - h.row);
      let c = h.col + dc, r = h.row + dr;
      const pathCells = [];
      while (!(c === CENTER.col && r === CENTER.row)) {
        pathCells.push({ col: c, row: r });
        c += dc; r += dr;
      }
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
      if (!canBlock) score -= 500;
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

    if (move.col === CENTER.col && move.row === CENTER.row) {
      pri = 1e9;
    } else {
      for (const { bs, filter } of APPROACH_CONFIGS) {
        if (move.col === bs.col && move.row === bs.row) {
          for (const h of snap.horses) {
            if (h.owner !== player || !filter(h)) continue;
            if (pathClear(h, snap.board)) { pri = Math.max(pri, 8000); }
          }
        }
      }
      for (const { bs, filter } of APPROACH_CONFIGS) {
        const horse = snap.horses.find(h => h.id === horseId);
        if (!horse || !filter(horse)) continue;
        if (!pathClear(horse, snap.board)) continue;
        const occupant = snap.board[boardKey(bs.col, bs.row)];
        if (occupant && occupant.owner === player) {
          pri = Math.max(pri, 7000);
        }
      }
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
      if (pri === 0 && (move.col === CENTER.col || move.row === CENTER.row)) {
        const dist = Math.abs(move.col - CENTER.col) + Math.abs(move.row - CENTER.row);
        pri = Math.max(pri, 100 + (10 - dist) * 10);
      }
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

let nodeCount = 0;

function minimax(snap, depth, alpha, beta, maximizing, deadline) {
  nodeCount++;

  if (nodeCount % 256 === 0 && Date.now() > deadline) {
    return maximizing ? alpha : beta;
  }

  if (snap.lastMoveToCenter) {
    const winner = 3 - snap.currentPlayer;
    return winner === aiPlayerNumber ? +50000 : -50000;
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

function getAIMove(snap) {
  const moves = generateOrderedMoves(snap);
  if (!moves.length) return null;

  const winMove = moves[0];
  if (winMove.move.col === CENTER.col && winMove.move.row === CENTER.row) {
    return { horseId: winMove.horseId, move: winMove.move };
  }

  const timeBudget = 12000;
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

    if (completed || depthBestVal > bestVal) {
      bestVal = depthBestVal;
      bestMove = depthBestMove;
    }

    self.postMessage({ type: 'log', msg: `[AI] depth=${depth} best=${depthBestVal.toFixed(0)} nodes=${nodeCount} ` +
      `move=${bestMove.horseId} -> (${bestMove.move.col},${bestMove.move.row})` });
  }

  return { horseId: bestMove.horseId, move: bestMove.move };
}

// ===== Message Handler =====

self.onmessage = function(e) {
  const { snap, nnWeights, playerNumber, forceHeuristic } = e.data;
  NNWeights = nnWeights || null;
  aiPlayerNumber = playerNumber;
  FORCE_HEURISTIC = forceHeuristic || false;

  const result = getAIMove(snap);
  self.postMessage({ type: 'result', result });
};

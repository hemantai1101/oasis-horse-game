'use strict';

// ===== AI State =====

const AIState = {
  enabled: false,
  playerNumber: 2,   // AI always plays as Black (Player 2)
  thinking: false,
};

// ===== Entry Points (called from game.js) =====

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

  const delay = 500 + Math.random() * 500;  // 500–1000 ms
  setTimeout(() => {
    if (GameState.phase !== 'playing') return;
    AIState.thinking = false;
    const result = getAIMove();
    if (result) executeMove(result.horse, result.move);
  }, delay);
}

// ===== AI Decision =====

function getAIMove() {
  const aiPlayer    = AIState.playerNumber;
  const humanPlayer = 3 - aiPlayer;

  // Gather all legal moves for AI pieces
  const candidates = [];
  for (const horse of GameState.horses) {
    if (horse.owner !== aiPlayer) continue;
    for (const move of getValidMoves(horse)) {
      candidates.push({ horse, move });
    }
  }
  if (!candidates.length) return null;

  // 1. Always take a winning move immediately
  const win = candidates.find(c => c.move.col === CENTER.col && c.move.row === CENTER.row);
  if (win) return win;

  // 2. Try to block opponent's winning slide path (75 % of the time — imperfect on purpose)
  if (Math.random() < 0.75) {
    const block = findBlockingMove(humanPlayer, candidates);
    if (block) return block;
  }

  // 3. Score remaining moves and pick from the top 3 with weighted randomness
  const scored = candidates.map(c => ({ ...c, score: scoreMove(c.horse, c.move) }));
  scored.sort((a, b) => b.score - a.score);
  return pickWeighted(scored.slice(0, 3));
}

// ===== Scoring =====

function scoreMove(horse, move) {
  const oldDist    = Math.abs(horse.col - CENTER.col) + Math.abs(horse.row - CENTER.row);
  const newDist    = Math.abs(move.col  - CENTER.col) + Math.abs(move.row  - CENTER.row);
  const improvement = oldDist - newDist;       // how much closer to center
  const destValue   = 22 - newDist;            // absolute closeness of destination
  const momentum    = Math.max(0, 10 - oldDist); // extra credit for already-close pieces
  const noise       = (Math.random() - 0.5) * 12; // controlled randomness

  return improvement * 12 + destValue * 1.5 + momentum * 2 + noise;
}

// ===== Blocking =====

// If the human has a slide path directly to center, find an AI move that
// places a piece along that path to interrupt it.
function findBlockingMove(humanPlayer, aiCandidates) {
  for (const horse of GameState.horses) {
    if (horse.owner !== humanPlayer) continue;
    const opponentMoves = getValidMoves(horse);
    const winnableSlides = opponentMoves.filter(
      m => m.col === CENTER.col && m.row === CENTER.row && m.moveType === 'slide'
    );
    for (const _slide of winnableSlides) {
      // Determine direction of the slide (cardinal only)
      const dc = horse.col === CENTER.col ? 0 : Math.sign(CENTER.col - horse.col);
      const dr = horse.row === CENTER.row ? 0 : Math.sign(CENTER.row - horse.row);
      let c = horse.col + dc, r = horse.row + dr;
      while (!(c === CENTER.col && r === CENTER.row)) {
        const block = aiCandidates.find(a => a.move.col === c && a.move.row === r);
        if (block) return block;
        c += dc; r += dr;
      }
    }
  }
  return null;
}

// ===== Weighted Random Pick =====

// Picks from a pre-sorted array using weights [6, 3, 1] so the best move
// is chosen most often but not always.
function pickWeighted(sorted) {
  if (!sorted.length) return null;
  const weights = [6, 3, 1].slice(0, sorted.length);
  const total   = weights.reduce((a, b) => a + b, 0);
  let rand = Math.random() * total;
  for (let i = 0; i < weights.length; i++) {
    rand -= weights[i];
    if (rand <= 0) return sorted[i];
  }
  return sorted[0];
}

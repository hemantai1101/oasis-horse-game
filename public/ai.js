'use strict';

// ===== Neural Network Model Loading =====
//
// Weights are loaded on the main thread so they can be transferred to the
// Web Worker on each move request.  The heavy minimax search runs entirely
// inside ai-worker.js, keeping the main thread (and UI) unblocked.

let NNWeights = null;

// Set to true to force heuristic evaluation even when weights are loaded.
const FORCE_HEURISTIC = true;

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

// ===== AI State =====

const AIState = {
  enabled: false,
  playerNumber: 2,   // AI always plays as Black (Player 2)
  thinking: false,
};

// ===== Web Worker =====

let _aiWorker = null;

function getAIWorker() {
  if (!_aiWorker) {
    _aiWorker = new Worker('ai-worker.js');
    _aiWorker.onmessage = function(e) {
      const { type, result, msg } = e.data;
      if (type === 'log') {
        console.log(msg);
        return;
      }
      // type === 'result'
      AIState.thinking = false;
      if (GameState.phase !== 'playing') { render(); return; }
      if (result) {
        const horse = GameState.horses.find(h => h.id === result.horseId);
        if (horse) executeMove(horse, result.move);
      }
    };
    _aiWorker.onerror = function(err) {
      console.error('[AI Worker error]', err);
      AIState.thinking = false;
      render();
    };
  }
  return _aiWorker;
}

// ===== Entry Points =====

function startVsComputer() {
  AIState.enabled = true;
  AIState.thinking = false;
  startGame();
}

// Called after each move; no-ops if it's not the AI's turn.
function scheduleAIMove() {
  if (!AIState.enabled) return;
  if (GameState.currentPlayer !== AIState.playerNumber) return;
  if (GameState.phase !== 'playing') return;

  AIState.thinking = true;
  render();  // show "thinking…" status immediately

  const snap = snapshotFromGameState();
  const delay = 400 + Math.random() * 300;
  setTimeout(() => {
    if (GameState.phase !== 'playing') { AIState.thinking = false; render(); return; }
    getAIWorker().postMessage({
      snap,
      nnWeights: NNWeights,
      playerNumber: AIState.playerNumber,
      forceHeuristic: FORCE_HEURISTIC,
    });
  }, delay);
}

// ===== State Snapshot (shared helper used by worker dispatch) =====

function snapshotFromGameState() {
  const horses = GameState.horses.map(h => ({ id: h.id, owner: h.owner, col: h.col, row: h.row }));
  const board = {};
  for (const h of horses) board[boardKey(h.col, h.row)] = h;
  return {
    horses, board,
    currentPlayer: GameState.currentPlayer,
    lastMoveToCenter: false,
    positionHistory: GameState.positionHistory ? GameState.positionHistory.slice() : [],
  };
}

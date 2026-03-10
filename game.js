'use strict';

// ===== Constants =====

const BOARD_SIZE = 11;
const CENTER = { col: 6, row: 6 };
const ROUNDS_TO_WIN = 2;

const ZONE = {
  DESERT: 'desert',
  FOREST: 'forest',
  OASIS:  'oasis',
};

const CORNER_CONFIGS = {
  TL: { positions: [[1,1],[2,1],[3,1],[1,2],[1,3]],       label: 'Top-Left'     },
  TR: { positions: [[11,1],[10,1],[9,1],[11,2],[11,3]],    label: 'Top-Right'    },
  BL: { positions: [[1,11],[2,11],[3,11],[1,10],[1,9]],    label: 'Bottom-Left'  },
  BR: { positions: [[11,11],[10,11],[9,11],[11,10],[11,9]],label: 'Bottom-Right' },
};

// Each entry: player 1 gets p1 corners, player 2 gets p2 corners
const DIAGONAL_PAIRS = [
  { p1: ['TL', 'BR'], p2: ['TR', 'BL'], label: 'Top-Left + Bottom-Right' },
  { p1: ['TR', 'BL'], p2: ['TL', 'BR'], label: 'Top-Right + Bottom-Left' },
];

const KNIGHT_OFFSETS = [
  [+2, +1], [+2, -1], [-2, +1], [-2, -1],
  [+1, +2], [+1, -2], [-1, +2], [-1, -2],
];

// ===== Game State =====

let GameState = {
  phase: 'corner_selection',  // 'corner_selection' | 'playing' | 'round_over' | 'match_over'
  scores: { 1: 0, 2: 0 },
  currentRound: 1,
  cornerChoice: null,   // 0 or 1 (index into DIAGONAL_PAIRS)
  board: {},            // "col,row" -> Horse object
  horses: [],           // all Horse objects
  currentPlayer: 1,
  selectedHorse: null,
  validMoves: [],       // [{col, row, moveType}]
  roundWinner: null,    // player number who won the last round
};

// ===== Utility Functions =====

function getZone(col, row) {
  const dist = Math.abs(col - CENTER.col) + Math.abs(row - CENTER.row);
  if (dist === 0) return ZONE.OASIS;
  if (dist <= 2)  return ZONE.FOREST;
  return ZONE.DESERT;
}

function isOnBoard(col, row) {
  return col >= 1 && col <= BOARD_SIZE && row >= 1 && row <= BOARD_SIZE;
}

function boardKey(col, row) {
  return `${col},${row}`;
}

function getHorseAt(col, row) {
  return GameState.board[boardKey(col, row)] || null;
}

function makeHorse(id, owner, col, row) {
  return { id, owner, col, row };
}

// ===== Move Calculation =====

function getSlideRay(horse, dc, dr) {
  // Slide to the end: horse travels until it hits a wall or another horse.
  // Only the final reachable cell in this direction is a valid destination.
  let lastCol = null;
  let lastRow = null;
  let c = horse.col + dc;
  let r = horse.row + dr;
  while (isOnBoard(c, r)) {
    if (getHorseAt(c, r)) break;  // blocked — stop before this cell
    lastCol = c;
    lastRow = r;
    c += dc;
    r += dr;
  }
  if (lastCol === null) return [];  // no room to slide in this direction
  return [{ col: lastCol, row: lastRow, moveType: 'slide' }];
}

function getSlideMoves(horse) {
  return [
    ...getSlideRay(horse,  1,  0),  // right
    ...getSlideRay(horse, -1,  0),  // left
    ...getSlideRay(horse,  0,  1),  // down
    ...getSlideRay(horse,  0, -1),  // up
  ];
}

function getKnightMoves(horse) {
  const moves = [];
  for (const [dc, dr] of KNIGHT_OFFSETS) {
    const c = horse.col + dc;
    const r = horse.row + dr;
    if (!isOnBoard(c, r)) continue;
    if (getHorseAt(c, r)) continue;
    if (getZone(c, r) !== ZONE.DESERT) continue;  // knights can only land on desert
    moves.push({ col: c, row: r, moveType: 'knight' });
  }
  return moves;
}

function getValidMoves(horse) {
  return [...getSlideMoves(horse), ...getKnightMoves(horse)];
}

// ===== Game Flow =====

function initGame() {
  GameState = {
    phase: 'corner_selection',
    scores: { 1: 0, 2: 0 },
    currentRound: 1,
    cornerChoice: null,
    board: {},
    horses: [],
    currentPlayer: 1,
    selectedHorse: null,
    validMoves: [],
    roundWinner: null,
  };
  render();
}

function handleCornerSelection(pairIndex) {
  GameState.cornerChoice = pairIndex;
  initRound();
}

function initRound() {
  GameState.phase = 'playing';
  GameState.board = {};
  GameState.horses = [];
  GameState.selectedHorse = null;
  GameState.validMoves = [];
  // Winner of previous round goes first; round 1 is always player 1
  if (GameState.roundWinner !== null) {
    GameState.currentPlayer = GameState.roundWinner;
  } else {
    GameState.currentPlayer = 1;
  }
  placeInitialHorses();
  render();
}

function placeInitialHorses() {
  const pair = DIAGONAL_PAIRS[GameState.cornerChoice];
  let p1id = 0;
  let p2id = 0;

  for (const cornerKey of pair.p1) {
    for (const [col, row] of CORNER_CONFIGS[cornerKey].positions) {
      const h = makeHorse(`p1_h${p1id++}`, 1, col, row);
      GameState.horses.push(h);
      GameState.board[boardKey(col, row)] = h;
    }
  }

  for (const cornerKey of pair.p2) {
    for (const [col, row] of CORNER_CONFIGS[cornerKey].positions) {
      const h = makeHorse(`p2_h${p2id++}`, 2, col, row);
      GameState.horses.push(h);
      GameState.board[boardKey(col, row)] = h;
    }
  }
}

function handleCellClick(col, row) {
  if (GameState.phase !== 'playing') return;

  const clickedHorse = getHorseAt(col, row);

  // Clicking own horse: select (or deselect if already selected)
  if (clickedHorse && clickedHorse.owner === GameState.currentPlayer) {
    if (GameState.selectedHorse && GameState.selectedHorse.id === clickedHorse.id) {
      deselectHorse();
    } else {
      selectHorse(clickedHorse);
    }
    return;
  }

  // Clicking a valid move destination
  if (GameState.selectedHorse) {
    const move = GameState.validMoves.find(m => m.col === col && m.row === row);
    if (move) {
      executeMove(GameState.selectedHorse, move);
      return;
    }
  }

  // Clicking anything else: deselect
  deselectHorse();
}

function selectHorse(horse) {
  GameState.selectedHorse = horse;
  GameState.validMoves = getValidMoves(horse);
  render();
}

function deselectHorse() {
  GameState.selectedHorse = null;
  GameState.validMoves = [];
  render();
}

function executeMove(horse, move) {
  // Update board
  delete GameState.board[boardKey(horse.col, horse.row)];
  horse.col = move.col;
  horse.row = move.row;
  GameState.board[boardKey(move.col, move.row)] = horse;

  // Check win condition
  if (move.col === CENTER.col && move.row === CENTER.row) {
    handleRoundWin(GameState.currentPlayer);
    return;
  }

  // Advance turn
  GameState.selectedHorse = null;
  GameState.validMoves = [];
  GameState.currentPlayer = GameState.currentPlayer === 1 ? 2 : 1;
  render();
}

function handleRoundWin(player) {
  GameState.scores[player]++;
  GameState.roundWinner = player;
  GameState.selectedHorse = null;
  GameState.validMoves = [];

  if (GameState.scores[player] >= ROUNDS_TO_WIN) {
    GameState.phase = 'match_over';
  } else {
    GameState.phase = 'round_over';
  }
  render();
}

function startNextRound() {
  GameState.currentRound++;
  initRound();
}

function resetMatch() {
  initGame();
}

// ===== Rendering =====

function render() {
  renderScoreBar();
  renderStatusBar();
  renderBoard();
  renderPhaseOverlay();
}

function renderScoreBar() {
  document.getElementById('score-p1').textContent = GameState.scores[1];
  document.getElementById('score-p2').textContent = GameState.scores[2];
  document.getElementById('round-indicator').textContent = `Round ${GameState.currentRound}`;
}

function renderStatusBar() {
  const el = document.getElementById('status-text');
  if (GameState.phase !== 'playing') {
    el.textContent = '';
    return;
  }
  const name = `Player ${GameState.currentPlayer}`;
  if (GameState.selectedHorse) {
    const parts = [];
    const slideCount  = GameState.validMoves.filter(m => m.moveType === 'slide').length;
    const knightCount = GameState.validMoves.filter(m => m.moveType === 'knight').length;
    if (slideCount)  parts.push(`${slideCount} slide (yellow — slides to end)`);
    if (knightCount) parts.push(`${knightCount} knight (purple — L-shape jump)`);
    const summary = parts.length ? parts.join(', ') : 'no valid moves';
    el.textContent = `${name}: ${summary} — tap a highlighted cell`;
  } else {
    el.textContent = `${name}'s turn — tap a horse to select it`;
  }
}

function renderBoard() {
  const boardEl = document.getElementById('board');
  boardEl.innerHTML = '';

  const selectedId   = GameState.selectedHorse ? GameState.selectedHorse.id : null;
  const validMoveSet = new Map(
    GameState.validMoves.map(m => [boardKey(m.col, m.row), m.moveType])
  );

  for (let row = 1; row <= BOARD_SIZE; row++) {
    for (let col = 1; col <= BOARD_SIZE; col++) {
      const cell = document.createElement('div');
      const zone = getZone(col, row);
      cell.className = `cell ${zone}`;

      const moveType = validMoveSet.get(boardKey(col, row));
      if (moveType) {
        cell.classList.add('valid-move', `valid-${moveType}`);
      }

      const horse = getHorseAt(col, row);
      if (horse) {
        const horseEl = document.createElement('div');
        horseEl.className = `horse player${horse.owner}`;
        if (horse.id === selectedId) {
          horseEl.classList.add('selected');
        }
        horseEl.textContent = '♞';
        cell.appendChild(horseEl);
      }

      // Capture col/row in closure
      const c = col;
      const r = row;
      cell.addEventListener('click', () => handleCellClick(c, r));

      boardEl.appendChild(cell);
    }
  }
}

function renderPhaseOverlay() {
  const overlay = document.getElementById('overlay');

  if (GameState.phase === 'corner_selection') {
    overlay.innerHTML = buildCornerSelectionHTML();
    overlay.classList.remove('hidden');
  } else if (GameState.phase === 'round_over') {
    overlay.innerHTML = buildRoundOverHTML();
    overlay.classList.remove('hidden');
  } else if (GameState.phase === 'match_over') {
    overlay.innerHTML = buildMatchOverHTML();
    overlay.classList.remove('hidden');
  } else {
    overlay.classList.add('hidden');
    return;
  }

  attachOverlayHandlers();
}

function buildCornerSelectionHTML() {
  const pair0 = DIAGONAL_PAIRS[0];
  const pair1 = DIAGONAL_PAIRS[1];

  function cornerNames(keys) {
    return keys.map(k => CORNER_CONFIGS[k].label).join(' &amp; ');
  }

  function p2Names(pairIdx) {
    return DIAGONAL_PAIRS[pairIdx].p2.map(k => CORNER_CONFIGS[k].label).join(' &amp; ');
  }

  return `
    <div class="overlay-panel">
      <h2>Choose Your Corners</h2>
      <p>Player 1, pick your two diagonal corners.<br>Player 2 gets the remaining pair.</p>
      <div class="corner-choices">
        <div class="corner-card" data-action="corner" data-pair="0">
          <h3>${cornerNames(pair0.p1)}</h3>
          <div class="corner-detail">
            <div class="p1-corners">Player 1: ${cornerNames(pair0.p1)}</div>
            <div class="p2-corners">Player 2: ${p2Names(0)}</div>
          </div>
        </div>
        <div class="corner-card" data-action="corner" data-pair="1">
          <h3>${cornerNames(pair1.p1)}</h3>
          <div class="corner-detail">
            <div class="p1-corners">Player 1: ${cornerNames(pair1.p1)}</div>
            <div class="p2-corners">Player 2: ${p2Names(1)}</div>
          </div>
        </div>
      </div>
    </div>
  `;
}

function buildRoundOverHTML() {
  const w = GameState.roundWinner;
  const wClass = `p${w}`;
  const nextPlayer = GameState.roundWinner; // winner goes first next round
  return `
    <div class="overlay-panel">
      <div class="winner-banner ${wClass}">Player ${w} wins the round!</div>
      <div class="score-display">
        <span class="p1-score">${GameState.scores[1]}</span>
        <span class="score-sep">–</span>
        <span class="p2-score">${GameState.scores[2]}</span>
      </div>
      <p>Player ${nextPlayer} goes first next round.</p>
      <button class="btn" data-action="next-round">Next Round</button>
    </div>
  `;
}

function buildMatchOverHTML() {
  const w = GameState.roundWinner;
  const wClass = `p${w}`;
  return `
    <div class="overlay-panel">
      <div class="winner-banner ${wClass}">Player ${w} wins the match!</div>
      <div class="score-display">
        <span class="p1-score">${GameState.scores[1]}</span>
        <span class="score-sep">–</span>
        <span class="p2-score">${GameState.scores[2]}</span>
      </div>
      <p>Congratulations to Player ${w}!</p>
      <button class="btn" data-action="reset">Play Again</button>
    </div>
  `;
}

function attachOverlayHandlers() {
  const overlay = document.getElementById('overlay');

  overlay.querySelectorAll('[data-action]').forEach(el => {
    el.addEventListener('click', () => {
      const action = el.dataset.action;
      if (action === 'corner') {
        handleCornerSelection(parseInt(el.dataset.pair, 10));
      } else if (action === 'next-round') {
        startNextRound();
      } else if (action === 'reset') {
        resetMatch();
      }
    });
  });
}

// ===== Entry Point =====

document.addEventListener('DOMContentLoaded', initGame);

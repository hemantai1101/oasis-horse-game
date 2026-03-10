'use strict';

// ===== Constants =====

const BOARD_SIZE = 11;
const CENTER = { col: 6, row: 6 };

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

const DIAGONAL_PAIRS = [
  { p1: ['TL', 'BR'], p2: ['TR', 'BL'], label: 'Top-Left + Bottom-Right' },
  { p1: ['TR', 'BL'], p2: ['TL', 'BR'], label: 'Top-Right + Bottom-Left' },
];

const KNIGHT_OFFSETS = [
  [+2, +1], [+2, -1], [-2, +1], [-2, -1],
  [+1, +2], [+1, -2], [-1, +2], [-1, -2],
];

// ===== Themes =====

const THEMES = {
  horse: {
    piece:        '♞',
    centerSymbol: null,
    goalName:     'Oasis',
    pieceName:    'horse',
    titleLabel:   'Oasis',
  },
  squirrel: {
    piece:        '🐿️',
    centerSymbol: '🌰',
    goalName:     'Acorn',
    pieceName:    'squirrel',
    titleLabel:   '🐿️ Squirrel',
  },
};

let currentTheme = 'horse';

function theme() {
  return THEMES[currentTheme];
}

function toggleTheme() {
  currentTheme = currentTheme === 'horse' ? 'squirrel' : 'horse';
  document.body.dataset.theme = currentTheme;
  document.getElementById('theme-btn').textContent =
    currentTheme === 'horse' ? '🐿️ Squirrel mode' : '♞ Horse mode';
  render();
}

// ===== Game State =====

let GameState = {
  phase: 'corner_selection',  // 'corner_selection' | 'playing' | 'game_over'
  cornerChoice: null,
  board: {},
  horses: [],
  currentPlayer: 1,
  selectedHorse: null,
  validMoves: [],
  winner: null,
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
  let lastCol = null;
  let lastRow = null;
  let c = horse.col + dc;
  let r = horse.row + dr;
  while (isOnBoard(c, r)) {
    if (getHorseAt(c, r)) break;
    lastCol = c;
    lastRow = r;
    c += dc;
    r += dr;
  }
  if (lastCol === null) return [];
  return [{ col: lastCol, row: lastRow, moveType: 'slide' }];
}

function getSlideMoves(horse) {
  return [
    ...getSlideRay(horse,  1,  0),
    ...getSlideRay(horse, -1,  0),
    ...getSlideRay(horse,  0,  1),
    ...getSlideRay(horse,  0, -1),
  ];
}

function getKnightMoves(horse) {
  const moves = [];
  for (const [dc, dr] of KNIGHT_OFFSETS) {
    const c = horse.col + dc;
    const r = horse.row + dr;
    if (!isOnBoard(c, r)) continue;
    if (getHorseAt(c, r)) continue;
    if (getZone(c, r) !== ZONE.DESERT) continue;
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
    cornerChoice: null,
    board: {},
    horses: [],
    currentPlayer: 1,
    selectedHorse: null,
    validMoves: [],
    winner: null,
  };
  render();
}

function handleCornerSelection(pairIndex) {
  GameState.cornerChoice = pairIndex;
  startGame();
}

function startGame() {
  GameState.phase = 'playing';
  GameState.board = {};
  GameState.horses = [];
  GameState.selectedHorse = null;
  GameState.validMoves = [];
  GameState.currentPlayer = 1;
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

  if (clickedHorse && clickedHorse.owner === GameState.currentPlayer) {
    if (GameState.selectedHorse && GameState.selectedHorse.id === clickedHorse.id) {
      deselectHorse();
    } else {
      selectHorse(clickedHorse);
    }
    return;
  }

  if (GameState.selectedHorse) {
    const move = GameState.validMoves.find(m => m.col === col && m.row === row);
    if (move) {
      executeMove(GameState.selectedHorse, move);
      return;
    }
  }

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
  delete GameState.board[boardKey(horse.col, horse.row)];
  horse.col = move.col;
  horse.row = move.row;
  GameState.board[boardKey(move.col, move.row)] = horse;

  if (move.col === CENTER.col && move.row === CENTER.row) {
    handleGameWin(GameState.currentPlayer);
    return;
  }

  GameState.selectedHorse = null;
  GameState.validMoves = [];
  GameState.currentPlayer = GameState.currentPlayer === 1 ? 2 : 1;
  render();
}

function handleGameWin(player) {
  GameState.winner = player;
  GameState.selectedHorse = null;
  GameState.validMoves = [];
  GameState.phase = 'game_over';
  render();
}

// ===== Rendering =====

function render() {
  renderStatusBar();
  renderBoard();
  renderPhaseOverlay();
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
    if (slideCount)  parts.push(`${slideCount} slide (yellow)`);
    if (knightCount) parts.push(`${knightCount} knight (purple)`);
    const summary = parts.length ? parts.join(', ') : 'no valid moves';
    el.textContent = `${name}: ${summary} — tap a highlighted cell`;
  } else {
    el.textContent = `${name}'s turn — tap a ${theme().pieceName} to select it`;
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

      // Center symbol (acorn etc.) shown when no piece is on the oasis
      if (zone === ZONE.OASIS && theme().centerSymbol && !getHorseAt(col, row)) {
        const sym = document.createElement('div');
        sym.className = 'center-symbol';
        sym.textContent = theme().centerSymbol;
        cell.appendChild(sym);
      }

      const horse = getHorseAt(col, row);
      if (horse) {
        const horseEl = document.createElement('div');
        horseEl.className = `horse player${horse.owner}`;
        if (horse.id === selectedId) horseEl.classList.add('selected');
        horseEl.textContent = theme().piece;
        cell.appendChild(horseEl);
      }

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
  } else if (GameState.phase === 'game_over') {
    overlay.innerHTML = buildGameOverHTML();
    overlay.classList.remove('hidden');
  } else {
    overlay.classList.add('hidden');
    return;
  }

  attachOverlayHandlers();
}

function buildCornerSelectionHTML() {
  function cornerNames(keys) {
    return keys.map(k => CORNER_CONFIGS[k].label).join(' &amp; ');
  }

  const t = theme();
  return `
    <div class="overlay-panel">
      <h2>Choose Your Corners</h2>
      <p>Player 1, pick your two diagonal corners.<br>Player 2 gets the remaining pair.<br>Race your ${t.pieceName}s to the ${t.goalName}!</p>
      <div class="corner-choices">
        ${DIAGONAL_PAIRS.map((pair, i) => `
          <div class="corner-card" data-action="corner" data-pair="${i}">
            <h3>${cornerNames(pair.p1)}</h3>
            <div class="corner-detail">
              <div class="p1-corners">Player 1: ${cornerNames(pair.p1)}</div>
              <div class="p2-corners">Player 2: ${cornerNames(pair.p2)}</div>
            </div>
          </div>
        `).join('')}
      </div>
    </div>
  `;
}

function buildGameOverHTML() {
  const w = GameState.winner;
  const t = theme();
  const centerDisplay = t.centerSymbol ? `${t.centerSymbol} ` : '';
  return `
    <div class="overlay-panel">
      <div class="winner-banner p${w}">Player ${w} wins!</div>
      <p>${centerDisplay}${t.goalName} reached — congratulations!</p>
      <button class="btn" data-action="play-again">Play Again</button>
    </div>
  `;
}

function attachOverlayHandlers() {
  const overlay = document.getElementById('overlay');
  overlay.querySelectorAll('[data-action]').forEach(el => {
    el.addEventListener('click', () => {
      const action = el.dataset.action;
      if (action === 'corner')      handleCornerSelection(parseInt(el.dataset.pair, 10));
      else if (action === 'play-again') initGame();
    });
  });
}

// ===== Entry Point =====

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('theme-btn').addEventListener('click', toggleTheme);
  initGame();
});

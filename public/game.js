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
  TL: { positions: [[1,1],[2,1],[3,1],[1,2],[1,3]]        },
  TR: { positions: [[11,1],[10,1],[9,1],[11,2],[11,3]]     },
  BL: { positions: [[1,11],[2,11],[3,11],[1,10],[1,9]]     },
  BR: { positions: [[11,11],[10,11],[9,11],[11,10],[11,9]] },
};

// Fixed corner assignment: White (P1) = TL+BR, Black (P2) = TR+BL
const PLAYER_CORNERS = {
  1: ['TL', 'BR'],
  2: ['TR', 'BL'],
};

const PLAYER_NAME = { 1: 'White', 2: 'Black' };

const KNIGHT_OFFSETS = [
  [+2, +1], [+2, -1], [-2, +1], [-2, -1],
  [+1, +2], [+1, -2], [-1, +2], [-1, -2],
];

const PIECE  = '🐿️';
const CENTER_SYMBOL = '🌰';

// ===== Game State =====

let GameState = {
  phase: 'start',   // 'start' | 'waiting' | 'joining' | 'playing' | 'game_over'
  board: {},
  horses: [],
  currentPlayer: 1,
  selectedHorse: null,
  validMoves: [],
  winner: null,
  totalMoves: 0,
  forfeitReason: null,  // 'timeout' | null
  lastMove: null,       // { fromCol, fromRow, toCol, toRow } | null
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
  let lastCol = null, lastRow = null;
  let c = horse.col + dc, r = horse.row + dr;
  while (isOnBoard(c, r)) {
    if (getHorseAt(c, r)) break;
    lastCol = c; lastRow = r;
    c += dc; r += dr;
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
    const c = horse.col + dc, r = horse.row + dr;
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

function initGame(isRestart = false) {
  // Exit any active multiplayer room first
  if (typeof exitRoom === 'function' && MultiplayerState.roomCode) {
    exitRoom();
  }
  if (isRestart) {
    posthog.capture('game_restarted', { from_phase: GameState.phase, moves_played: GameState.totalMoves });
  }
  AIState.enabled  = false;
  AIState.thinking = false;
  GameState = {
    phase: 'start',
    board: {},
    horses: [],
    currentPlayer: 1,
    selectedHorse: null,
    validMoves: [],
    winner: null,
    totalMoves: 0,
    forfeitReason: null,
    lastMove: null,
  };
  render();
}

function startGame() {
  posthog.capture('game_started');
  GameState.phase = 'playing';
  GameState.board = {};
  GameState.horses = [];
  GameState.selectedHorse = null;
  GameState.validMoves = [];
  GameState.currentPlayer = 1;
  GameState.winner = null;
  GameState.forfeitReason = null;
  GameState.lastMove = null;
  placeInitialHorses();
  render();
}

function placeInitialHorses() {
  for (const player of [1, 2]) {
    let id = 0;
    for (const cornerKey of PLAYER_CORNERS[player]) {
      for (const [col, row] of CORNER_CONFIGS[cornerKey].positions) {
        const h = makeHorse(`p${player}_h${id++}`, player, col, row);
        GameState.horses.push(h);
        GameState.board[boardKey(col, row)] = h;
      }
    }
  }
}

function handleCellClick(col, row) {
  if (GameState.phase !== 'playing') return;

  // Block clicks while the computer is deciding
  if (AIState.thinking) return;
  // Block clicks on the computer's turn
  if (AIState.enabled && GameState.currentPlayer === AIState.playerNumber) return;

  // Multiplayer: block interaction when it's not your turn
  if (MultiplayerState.roomCode && GameState.currentPlayer !== MultiplayerState.myPlayerNumber) return;

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
  posthog.capture('piece_selected', {
    player: PLAYER_NAME[horse.owner],
    from: { col: horse.col, row: horse.row },
    slide_moves: GameState.validMoves.filter(m => m.moveType === 'slide').length,
    knight_moves: GameState.validMoves.filter(m => m.moveType === 'knight').length,
  });
  render();
}

function deselectHorse() {
  GameState.selectedHorse = null;
  GameState.validMoves = [];
  render();
}

function executeMove(horse, move) {
  const fromCol = horse.col, fromRow = horse.row;
  GameState.lastMove = { fromCol, fromRow, toCol: move.col, toRow: move.row };
  delete GameState.board[boardKey(horse.col, horse.row)];
  horse.col = move.col;
  horse.row = move.row;
  GameState.board[boardKey(move.col, move.row)] = horse;
  GameState.totalMoves++;

  posthog.capture('move_executed', {
    player: PLAYER_NAME[GameState.currentPlayer],
    move_type: move.moveType,
    from: { col: fromCol, row: fromRow },
    to: { col: move.col, row: move.row },
    move_number: GameState.totalMoves,
  });

  if (move.col === CENTER.col && move.row === CENTER.row) {
    GameState.winner = GameState.currentPlayer;
    GameState.selectedHorse = null;
    GameState.validMoves = [];
    GameState.phase = 'game_over';
    posthog.capture('game_over', {
      winner: PLAYER_NAME[GameState.winner],
      total_moves: GameState.totalMoves,
    });
    if (MultiplayerState.roomCode) pushGameState(null);
    render();
    return;
  }

  GameState.selectedHorse = null;
  GameState.validMoves = [];
  GameState.currentPlayer = GameState.currentPlayer === 1 ? 2 : 1;
  if (MultiplayerState.roomCode) pushGameState(null);
  scheduleAIMove();
  render();
}

// ===== Rendering =====

function render() {
  renderStatusBar();
  renderBoard();
  renderPhaseOverlay();
  if (typeof updateTimerUI === 'function') updateTimerUI();
}

function renderStatusBar() {
  const el = document.getElementById('status-text');
  if (GameState.phase !== 'playing') { el.innerHTML = ''; return; }

  const name = PLAYER_NAME[GameState.currentPlayer];
  const cls  = `turn-dot p${GameState.currentPlayer}`;

  if (AIState.enabled && AIState.thinking) {
    el.innerHTML = `<span class="${cls}"></span>Computer is thinking…`;
    return;
  }

  // Show "(You)" label in multiplayer so players know which colour they are
  const youLabel = MultiplayerState.roomCode
    ? (GameState.currentPlayer === MultiplayerState.myPlayerNumber ? ' <em>(You)</em>' : '')
    : '';

  if (GameState.selectedHorse) {
    const slideCount  = GameState.validMoves.filter(m => m.moveType === 'slide').length;
    const knightCount = GameState.validMoves.filter(m => m.moveType === 'knight').length;
    const parts = [];
    if (slideCount)  parts.push(`${slideCount} slide`);
    if (knightCount) parts.push(`${knightCount} knight`);
    const summary = parts.length ? parts.join(', ') : 'no valid moves';
    el.innerHTML = `<span class="${cls}"></span>${name}${youLabel}: ${summary} — tap a highlighted cell`;
  } else {
    el.innerHTML = `<span class="${cls}"></span>${name}${youLabel}'s turn — tap a squirrel to select`;
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
      cell.className = `cell ${getZone(col, row)}`;

      const moveType = validMoveSet.get(boardKey(col, row));
      if (moveType) cell.classList.add('valid-move', `valid-${moveType}`);

      const lm = GameState.lastMove;
      if (lm) {
        if (col === lm.fromCol && row === lm.fromRow) cell.classList.add('last-move-from');
        if (col === lm.toCol   && row === lm.toRow)   cell.classList.add('last-move-to');
      }

      const zone = getZone(col, row);
      if (zone === ZONE.OASIS && !getHorseAt(col, row)) {
        const sym = document.createElement('div');
        sym.className = 'center-symbol';
        sym.textContent = CENTER_SYMBOL;
        cell.appendChild(sym);
      }

      const horse = getHorseAt(col, row);
      if (horse) {
        const horseEl = document.createElement('div');
        horseEl.className = `horse player${horse.owner}`;
        if (horse.id === selectedId) horseEl.classList.add('selected');
        horseEl.textContent = PIECE;
        cell.appendChild(horseEl);
      }

      const c = col, r = row;
      cell.addEventListener('click', () => handleCellClick(c, r));
      boardEl.appendChild(cell);
    }
  }
}

function renderPhaseOverlay() {
  const overlay = document.getElementById('overlay');

  if (GameState.phase === 'start') {
    overlay.innerHTML = buildStartHTML();
    overlay.classList.remove('hidden');
  } else if (GameState.phase === 'waiting') {
    overlay.innerHTML = typeof buildWaitingHTML === 'function' ? buildWaitingHTML() : '<div class="overlay-panel"><p>Loading…</p></div>';
    overlay.classList.remove('hidden');
    // Wire copy button (special handler, not data-action)
    const copyBtn = document.getElementById('copy-code-btn');
    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(MultiplayerState.roomCode || '').then(() => {
          copyBtn.textContent = '✓';
          setTimeout(() => { copyBtn.textContent = '📋'; }, 1500);
        }).catch(() => {
          // fallback: select text
          const codeEl = document.getElementById('room-code-text');
          if (codeEl) {
            const range = document.createRange();
            range.selectNode(codeEl);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
          }
        });
      });
    }
  } else if (GameState.phase === 'joining') {
    overlay.innerHTML = typeof buildJoinHTML === 'function' ? buildJoinHTML() : '<div class="overlay-panel"><p>Loading…</p></div>';
    overlay.classList.remove('hidden');
    setTimeout(() => {
      const input = document.getElementById('room-code-input');
      if (input) {
        input.focus();
        input.addEventListener('input', () => { input.value = input.value.toUpperCase(); });
        input.addEventListener('keydown', e => { if (e.key === 'Enter') joinRoom(input.value); });
      }
    }, 50);
  } else if (GameState.phase === 'game_over') {
    overlay.innerHTML = buildGameOverHTML();
    overlay.classList.remove('hidden');
  } else {
    overlay.classList.add('hidden');
    return;
  }

  overlay.querySelectorAll('[data-action]').forEach(el => {
    el.addEventListener('click', () => handleOverlayAction(el.dataset.action));
  });
}

function handleOverlayAction(action) {
  switch (action) {
    case 'start':
      startGame();
      break;
    case 'start-vs-computer':
      startVsComputer();
      break;
    case 'again':
      initGame();
      break;
    case 'again-vs-computer':
      startVsComputer();
      break;
    case 'create-room':
      if (typeof createRoom === 'function') createRoom();
      break;
    case 'join-room-prompt':
      GameState.phase = 'joining';
      render();
      break;
    case 'join-room': {
      const input = document.getElementById('room-code-input');
      if (input && typeof joinRoom === 'function') joinRoom(input.value);
      break;
    }
    case 'cancel-room':
      if (typeof exitRoom === 'function') exitRoom();
      initGame();
      break;
  }
}

function buildStartHTML() {
  const mpAvailable = window.db != null;
  const mpButtons = mpAvailable ? `
    <button class="btn btn-secondary" data-action="create-room">Create Online Room</button>
    <button class="btn btn-secondary" data-action="join-room-prompt">Join a Room</button>
  ` : '';
  return `
    <div class="overlay-panel">
      <h2>🐿️ Squirrel</h2>
      <div class="mode-buttons">
        <button class="btn" data-action="start">Play Locally</button>
        <button class="btn" data-action="start-vs-computer">vs Computer</button>
        ${mpButtons}
      </div>
    </div>
  `;
}

function buildGameOverHTML() {
  const w    = GameState.winner;
  const isMP = MultiplayerState.roomCode !== null;

  let winnerLabel;
  if (isMP) {
    winnerLabel = PLAYER_NAME[w];
  } else if (AIState.enabled) {
    winnerLabel = w === AIState.playerNumber ? 'Computer' : 'You';
  } else {
    winnerLabel = PLAYER_NAME[w];
  }

  let subtitle = '';
  if (GameState.forfeitReason === 'timeout') {
    const loser = w === 1 ? 2 : 1;
    subtitle = `<p class="forfeit-reason">${PLAYER_NAME[loser]} ran out of time</p>`;
  }

  const btnAction = isMP ? 'cancel-room' : (AIState.enabled ? 'again-vs-computer' : 'again');
  const btnLabel  = isMP ? 'Main Menu'   : 'Play Again';

  return `
    <div class="overlay-panel">
      <div class="winner-banner p${w}">${winnerLabel} wins!</div>
      ${subtitle}
      <button class="btn" data-action="${btnAction}">${btnLabel}</button>
    </div>
  `;
}

function showRulesModal() {
  posthog.capture('rules_opened');
  const modal = document.getElementById('rules-modal');
  modal.classList.remove('hidden');
}

function hideRulesModal() {
  document.getElementById('rules-modal').classList.add('hidden');
}

// ===== Entry Point =====

document.addEventListener('DOMContentLoaded', () => {
  // ===== Theme =====
  function setTheme(name) {
    document.documentElement.setAttribute('data-theme', name);
    localStorage.setItem('theme', name);
    document.getElementById('theme-btn').textContent = name === 'dark' ? '☀️' : '🌙';
  }
  setTheme(localStorage.getItem('theme') || 'dark');
  document.getElementById('theme-btn').addEventListener('click', () => {
    setTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  });

  document.getElementById('restart-btn').addEventListener('click', () => initGame(true));
  document.getElementById('rules-btn').addEventListener('click', showRulesModal);
  document.getElementById('rules-close').addEventListener('click', hideRulesModal);
  document.getElementById('rules-modal').addEventListener('click', e => {
    if (e.target === e.currentTarget) hideRulesModal();
  });
  initGame();
  if (typeof tryReconnect === 'function') tryReconnect();

  // Re-render on resize so the board reflows to the new viewport width.
  // CSS vw-based --cell-size auto-updates, but a render() call ensures the
  // grid DOM is fresh and any layout quirks are resolved.
  let _resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(render, 50);
  });
});

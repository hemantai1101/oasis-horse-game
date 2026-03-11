'use strict';

// ===== Multiplayer State =====

const TURN_LIMIT_SECONDS = 30;
const ROOM_CODE_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';

let MultiplayerState = {
  roomCode:       null,   // null = local/offline mode
  myPlayerNumber: null,   // 1 or 2
  myUid:          null,   // Firebase Auth UID
  status:         'idle', // 'idle' | 'waiting' | 'active'
  timerInterval:  null,
  timerSecondsLeft: 30,
  roomListener:   null,   // fn to unsubscribe from Firebase listener
};

// ===== Session Identity =====

async function getAuthenticatedUid() {
  if (!window.auth) throw new Error('Firebase Auth not initialized.');
  if (window.auth.currentUser) {
    return window.auth.currentUser.uid;
  }
  try {
    const userCredential = await window.auth.signInAnonymously();
    return userCredential.user.uid;
  } catch (error) {
    console.error("Anonymous sign-in failed:", error);
    throw new Error('Could not sign in to multiplayer.');
  }
}

// ===== Room Code Generation =====

function generateRoomCode() {
  let code = '';
  for (let i = 0; i < 5; i++) {
    code += ROOM_CODE_CHARS[Math.floor(Math.random() * ROOM_CODE_CHARS.length)];
  }
  return code;
}

async function findUniqueRoomCode() {
  for (let i = 0; i < 10; i++) {
    const code = generateRoomCode();
    const snap = await window.db.ref(`rooms/${code}`).get();
    if (!snap.exists() || snap.val().status === 'done') return code;
  }
  throw new Error('Could not generate a unique room code after 10 attempts');
}

// ===== Create Room =====

async function createRoom() {
  if (!window.db) { alert('Multiplayer unavailable — Firebase not configured.'); return; }
  try {
    const [code, uid] = await Promise.all([findUniqueRoomCode(), getAuthenticatedUid()]);
    await window.db.ref(`rooms/${code}`).set({
      created: Date.now(),
      status:  'waiting',
      p1uid:   uid,
      p2uid:   null,
      game:    null,
    });
    MultiplayerState.roomCode       = code;
    MultiplayerState.myPlayerNumber = 1;
    MultiplayerState.myUid          = uid;
    MultiplayerState.status         = 'waiting';
    sessionStorage.setItem('mp_room', JSON.stringify({ roomCode: code, playerNumber: 1, uid }));
    listenToRoom(code);
    GameState.phase = 'waiting';
    render();
  } catch (e) {
    console.error('createRoom error:', e);
    alert('Could not create room. Check your connection and try again.');
  }
}

// ===== Join Room =====

async function joinRoom(rawCode) {
  if (!window.db) { alert('Multiplayer unavailable — Firebase not configured.'); return; }
  const code = (rawCode || '').trim().toUpperCase();
  if (code.length !== 5) {
    showJoinError('Please enter a 5-character room code.');
    return;
  }

  const joinBtn = document.getElementById('join-btn');
  if (joinBtn) { joinBtn.disabled = true; joinBtn.textContent = 'Joining…'; }

  try {
    const uid = await getAuthenticatedUid();
    const snap = await window.db.ref(`rooms/${code}`).get();
    if (!snap.exists()) {
      showJoinError('Room not found. Check the code and try again.'); return;
    }
    const room = snap.val();
    if (room.status !== 'waiting') {
      showJoinError(room.status === 'active' ? 'Room is already full.' : 'Room is no longer active.'); return;
    }

    await window.db.ref(`rooms/${code}`).update({ p2uid: uid, status: 'active' });

    MultiplayerState.roomCode       = code;
    MultiplayerState.myPlayerNumber = 2;
    MultiplayerState.myUid          = uid;
    MultiplayerState.status         = 'active';
    sessionStorage.setItem('mp_room', JSON.stringify({ roomCode: code, playerNumber: 2, uid }));

    // Show "starting…" while P1 initialises the game state
    GameState.phase = 'waiting';
    render();
    listenToRoom(code);
  } catch (e) {
    console.error('joinRoom error:', e);
    showJoinError('Connection failed. Please try again.');
  } finally {
    if (joinBtn) { joinBtn.disabled = false; joinBtn.textContent = 'Join'; }
  }
}

function showJoinError(msg) {
  const el = document.getElementById('join-error');
  if (el) el.textContent = msg;
}

// ===== Firebase Listener =====

function listenToRoom(code) {
  if (MultiplayerState.roomListener) MultiplayerState.roomListener();
  const ref = window.db.ref(`rooms/${code}`);
  const handler = ref.on('value', snap => {
    if (!snap.exists()) return;
    applyRemoteState(snap.val());
  });
  MultiplayerState.roomListener = () => ref.off('value', handler);
}

// ===== Apply Remote State =====

function applyRemoteState(room) {
  // P2 just joined — P1 initialises the board and pushes it to the DB
  if (room.status === 'active' &&
      MultiplayerState.status === 'waiting' &&
      MultiplayerState.myPlayerNumber === 1) {
    MultiplayerState.status = 'active';
    startMultiplayerGame();
    return;
  }

  if (!room.game) return;

  const prevPlayer = GameState.currentPlayer;
  const g = room.game;

  // Rebuild GameState from the remote snapshot
  GameState.horses = g.horses.map(h => ({ id: h.id, owner: h.owner, col: h.col, row: h.row }));
  GameState.board  = {};
  for (const h of GameState.horses) {
    GameState.board[boardKey(h.col, h.row)] = h;
  }
  GameState.currentPlayer  = g.currentPlayer;
  GameState.totalMoves     = g.totalMoves;
  GameState.winner         = g.winner  || null;
  GameState.phase          = g.phase;
  GameState.forfeitReason  = g.forfeitReason || null;
  GameState.selectedHorse  = null;
  GameState.validMoves     = [];
  GameState.lastMove       = g.lastMove || null;

  if (GameState.phase === 'playing') {
    if (GameState.currentPlayer !== prevPlayer || !MultiplayerState.timerInterval) {
      startMpTimer(g.lastMoveAt);
    }
  } else {
    stopMpTimer();
  }

  render();
}

function startMultiplayerGame() {
  // Only P1 calls this once P2 joins
  startGame();          // from game.js — resets GameState and calls placeInitialHorses
  pushGameState(null);  // push initial board to DB so P2 can see it
}

// ===== Push State to DB =====

function pushGameState(forfeitReason) {
  if (!MultiplayerState.roomCode || !window.db) return;
  const update = {};
  update[`rooms/${MultiplayerState.roomCode}/game`] = {
    phase:         GameState.phase,
    currentPlayer: GameState.currentPlayer,
    totalMoves:    GameState.totalMoves,
    winner:        GameState.winner,
    forfeitReason: forfeitReason || null,
    horses:        GameState.horses.map(h => ({ id: h.id, owner: h.owner, col: h.col, row: h.row })),
    lastMove:      GameState.lastMove || null,
    lastMoveAt:    Date.now(),
  };
  if (GameState.phase === 'game_over') {
    update[`rooms/${MultiplayerState.roomCode}/status`] = 'done';
  }
  return window.db.ref().update(update).catch(e => console.error('pushGameState error:', e));
}

// ===== Turn Timer =====

function startMpTimer(lastMoveAt) {
  stopMpTimer();
  const elapsed = Math.floor((Date.now() - (lastMoveAt || Date.now())) / 1000);
  MultiplayerState.timerSecondsLeft = Math.max(0, TURN_LIMIT_SECONDS - elapsed);
  updateTimerUI();

  MultiplayerState.timerInterval = setInterval(() => {
    MultiplayerState.timerSecondsLeft = Math.max(0, MultiplayerState.timerSecondsLeft - 1);
    updateTimerUI();
    if (MultiplayerState.timerSecondsLeft <= 0) {
      stopMpTimer();
      handleTimerExpiry();
    }
  }, 1000);
}

function stopMpTimer() {
  if (MultiplayerState.timerInterval) {
    clearInterval(MultiplayerState.timerInterval);
    MultiplayerState.timerInterval = null;
  }
}

function handleTimerExpiry() {
  if (GameState.phase !== 'playing') return;
  // Both clients will write the same outcome; first write wins
  GameState.winner       = GameState.currentPlayer === 1 ? 2 : 1;
  GameState.phase        = 'game_over';
  GameState.forfeitReason = 'timeout';
  pushGameState('timeout');
  render();
}

function updateTimerUI() {
  const bar = document.getElementById('timer-bar');
  if (!bar) return;

  if (!MultiplayerState.roomCode || GameState.phase !== 'playing') {
    bar.classList.add('hidden');
    return;
  }

  bar.classList.remove('hidden');
  const isMyTurn = GameState.currentPlayer === MultiplayerState.myPlayerNumber;
  bar.classList.toggle('my-turn', isMyTurn);

  const pct   = (MultiplayerState.timerSecondsLeft / TURN_LIMIT_SECONDS) * 100;
  const fill  = document.getElementById('timer-fill');
  const count = document.getElementById('timer-count');
  const label = document.getElementById('timer-label');

  if (fill) {
    fill.style.width = pct + '%';
    fill.classList.toggle('timer-danger',  pct <= 25);
    fill.classList.toggle('timer-warning', pct > 25 && pct <= 50);
    fill.classList.remove(pct > 50 ? 'timer-warning' : '', pct > 50 ? 'timer-danger' : '');
  }
  if (count) count.textContent = MultiplayerState.timerSecondsLeft;
  if (label) label.textContent = isMyTurn ? 'Your turn' : (PLAYER_NAME[GameState.currentPlayer] + '\'s turn');
}

// ===== Exit Room =====

function exitRoom() {
  stopMpTimer();
  if (MultiplayerState.roomListener) {
    MultiplayerState.roomListener();
    MultiplayerState.roomListener = null;
  }
  sessionStorage.removeItem('mp_room');
  Object.assign(MultiplayerState, {
    roomCode: null, myPlayerNumber: null, myUid: null,
    status: 'idle', timerSecondsLeft: 30,
  });
  const bar = document.getElementById('timer-bar');
  if (bar) bar.classList.add('hidden');
}

// ===== Overlay HTML Builders =====

function buildWaitingHTML() {
  const code    = MultiplayerState.roomCode;
  const waiting = MultiplayerState.myPlayerNumber === 1 && MultiplayerState.status === 'waiting';

  if (waiting) {
    return `
      <div class="overlay-panel">
        <h2>Waiting for opponent…</h2>
        <div class="room-code-display">
          <span class="room-code-label">Room Code</span>
          <span class="room-code-value" id="room-code-text">${code}</span>
          <button class="copy-btn" id="copy-code-btn" title="Copy code">📋</button>
        </div>
        <p class="room-hint">Share this code with your opponent</p>
        <button class="btn btn-ghost" data-action="cancel-room">Cancel</button>
      </div>
    `;
  }

  // P2 waiting for P1 to push the initial board
  return `
    <div class="overlay-panel">
      <h2>Joined room <span class="room-code-value-sm">${code}</span></h2>
      <p class="room-hint">Starting game…</p>
    </div>
  `;
}

function buildJoinHTML() {
  return `
    <div class="overlay-panel">
      <h2>Join a Game</h2>
      <div class="join-input-row">
        <input
          id="room-code-input"
          class="room-code-input"
          type="text"
          maxlength="5"
          placeholder="XXXXX"
          autocomplete="off"
          autocapitalize="characters"
          spellcheck="false"
        />
      </div>
      <span id="join-error" class="join-error"></span>
      <div class="join-actions">
        <button id="join-btn" class="btn" data-action="join-room">Join</button>
        <button class="btn btn-ghost" data-action="cancel-room">Cancel</button>
      </div>
    </div>
  `;
}

// ===== Session Reconnect =====

function tryReconnect() {
  if (!window.db) return;
  const saved = sessionStorage.getItem('mp_room');
  if (!saved) return;
  getAuthenticatedUid().then(uid => {
    try {
      const { roomCode, playerNumber } = JSON.parse(saved);
      window.db.ref(`rooms/${roomCode}`).get().then(snap => {
        if (!snap.exists() || snap.val().status === 'done') {
          sessionStorage.removeItem('mp_room');
          return;
        }
        const room = snap.val();
        // Verify this user is still a player in this room
        if (uid !== room.p1uid && uid !== room.p2uid) {
          sessionStorage.removeItem('mp_room');
          return;
        }
        MultiplayerState.roomCode       = roomCode;
        MultiplayerState.myPlayerNumber = playerNumber;
        MultiplayerState.myUid          = uid;
        MultiplayerState.status         = room.status;
        if (room.status === 'waiting') {
          GameState.phase = 'waiting';
          render();
        }
        listenToRoom(roomCode);
      });
    } catch (e) {
      sessionStorage.removeItem('mp_room');
    }
  }).catch(e => {
    console.error('Reconnect failed:', e);
    sessionStorage.removeItem('mp_room');
  });
}

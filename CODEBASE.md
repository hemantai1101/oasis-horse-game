# Codebase Reference

Quick reference for AI assistants. Read this before making changes.

---

## File Map

```
public/
  index.html       — HTML structure and DOM skeleton
  style.css        — All styling (CSS variables, layout, components)
  game.js          — Core game logic, state, rendering
  multiplayer.js   — Firebase multiplayer, room management, turn timer
  ai.js            — AI opponent (neural network + minimax)
  firebase-init.js — Firebase app/db/auth initialization (window.db, window.auth)
  model/           — Pretrained NN weights for AI
```

---

## HTML Structure (`index.html`)

```
#app
  #score-bar          — Header: title + Restart / Theme / Rules buttons
  #status-bar         — Turn indicator text; also contains #timer-bar as second row
    #status-text      — Rendered by renderStatusBar() in game.js
    #timer-bar        — Multiplayer countdown (hidden via opacity when not in MP)
      #timer-label    — "Your turn" / "White's turn"
      #timer-track > #timer-fill  — Progress bar
      #timer-count    — Seconds remaining
  #game-area          — position:relative wrapper
    #board            — 11×11 CSS Grid, cells built dynamically
    #overlay          — Phase overlays (start / waiting / joining / game_over)
  #legend             — Slide / Knight move key

#rules-modal          — Fixed rules panel (outside #app)
```

**Key DOM IDs:** `status-text`, `timer-bar`, `timer-fill`, `timer-count`, `timer-label`, `board`, `overlay`

---

## Game State (`game.js`)

```js
GameState = {
  phase: 'start',        // 'start' | 'waiting' | 'joining' | 'playing' | 'game_over'
  board: {},             // { "col,row": horseObj } — fast lookup
  horses: [],            // all horse objects [{ id, owner, col, row }]
  currentPlayer: 1,      // 1 = White, 2 = Black
  selectedHorse: null,   // currently selected horse object or null
  validMoves: [],        // [{ col, row, moveType: 'slide'|'knight' }]
  winner: null,          // 1 | 2 | null
  totalMoves: 0,
  forfeitReason: null,   // 'timeout' | null
  lastMove: null,        // { fromCol, fromRow, toCol, toRow } | null
}
```

### Board

- **11×11 grid**, 1-indexed (`col` 1–11, `row` 1–11)
- **Center/goal:** `col:6, row:6` (🌰)
- **Zones** (Manhattan distance from center):
  - `dist === 0` → oasis (win cell)
  - `dist <= 2`  → forest (slide only)
  - `dist > 2`   → desert (slide + knight)
- **Board lookup:** `GameState.board["col,row"]` → horse object or undefined

### Starting positions

| Player | Corners |
|--------|---------|
| 1 (White) | TL `[1,1][2,1][3,1][1,2][1,3]` + BR `[11,11][10,11][9,11][11,10][11,9]` |
| 2 (Black) | TR `[11,1][10,1][9,1][11,2][11,3]` + BL `[1,11][2,11][3,11][1,10][1,9]` |

### Move rules

- **Slide:** Straight line (4 directions). Slides until board edge or blocking piece; lands on last empty cell.
- **Knight:** L-shape (8 offsets). Only allowed if destination is in the **desert** zone. Cannot land on occupied cells.

### Key functions

| Function | File | Purpose |
|----------|------|---------|
| `initGame(isRestart?)` | game.js | Resets all state, exits MP room if active, calls `render()` |
| `startGame()` | game.js | Sets phase to `'playing'`, places horses, renders |
| `handleCellClick(col, row)` | game.js | Main input handler — select piece or execute move |
| `executeMove(horse, move)` | game.js | Moves piece, checks win, advances turn, pushes to DB in MP |
| `getValidMoves(horse)` | game.js | Returns slide + knight moves for a horse |
| `render()` | game.js | Calls `renderStatusBar()`, `renderBoard()`, `renderPhaseOverlay()`, `updateTimerUI()` |
| `renderPhaseOverlay()` | game.js | Shows/hides `#overlay` based on `GameState.phase` |
| `buildStartHTML()` | game.js | Start screen HTML — shows MP buttons only if `window.db != null` |
| `buildGameOverHTML()` | game.js | Game over HTML — adapts text for MP / AI / local |

### Multiplayer guards in game.js

```js
// Block interaction on opponent's turn:
if (MultiplayerState.roomCode && GameState.currentPlayer !== MultiplayerState.myPlayerNumber) return;

// Push state after every move:
if (MultiplayerState.roomCode) pushGameState(null);
```

---

## Multiplayer State (`multiplayer.js`)

```js
MultiplayerState = {
  roomCode: null,         // null = local; string = in MP room
  myPlayerNumber: null,   // 1 or 2
  myUid: null,            // Firebase anonymous Auth UID
  status: 'idle',         // 'idle' | 'waiting' | 'active'
  timerInterval: null,
  timerSecondsLeft: 30,
  roomListener: null,     // unsubscribe fn for Firebase .on()
}
```

**Detect multiplayer anywhere:** `MultiplayerState.roomCode !== null`

### Room lifecycle

1. **P1** calls `createRoom()` → writes `rooms/{code}` to Firebase, sets `status:'waiting'`, calls `listenToRoom()`
2. **P2** calls `joinRoom(code)` → updates room to `status:'active'`, calls `listenToRoom()`
3. **P1** detects P2 joined via `applyRemoteState()` → calls `startMultiplayerGame()` → `startGame()` + `pushGameState()`
4. After each move: `pushGameState()` writes updated `GameState` to `rooms/{code}/game`
5. Both clients listen via `listenToRoom()` → `applyRemoteState()` rebuilds local `GameState` from DB snapshot

### Firebase DB schema

```
rooms/{code}/
  created: timestamp
  status: 'waiting' | 'active' | 'done'
  p1uid: string
  p2uid: string | null
  game: {
    phase, currentPlayer, totalMoves, winner, forfeitReason,
    horses: [{ id, owner, col, row }],
    lastMove: { fromCol, fromRow, toCol, toRow } | null,
    lastMoveAt: timestamp,
  }
```

### Key functions

| Function | Purpose |
|----------|---------|
| `createRoom()` | Generates code, writes to Firebase, starts listener |
| `joinRoom(code)` | Validates + joins room, starts listener |
| `listenToRoom(code)` | Firebase `.on('value')` → calls `applyRemoteState()` on changes |
| `applyRemoteState(room)` | Rebuilds `GameState` from DB snapshot, starts/stops timer |
| `pushGameState(forfeitReason)` | Writes current `GameState` to DB |
| `exitRoom()` | Clears `MultiplayerState`, unsubscribes listener, hides timer |
| `tryReconnect()` | On page load: checks `localStorage('mp_room')`, reconnects if < 4h old |
| `updateTimerUI()` | Hides timer if not MP; shows/updates fill bar + count. Called by `render()` |
| `startMpTimer(lastMoveAt)` | Starts 30s countdown interval |
| `handleTimerExpiry()` | Sets winner to opponent, phase `'game_over'`, calls `pushGameState('timeout')` |

---

## Timer Layout

The `#timer-bar` lives **inside** `#status-bar` as a `position: absolute; bottom: 0` strip.

- **Design:** 4px thin red bar sitting at the very bottom edge of `#status-bar` (visually between header and game board)
- **Hidden (non-MP):** `display: none` — absolutely positioned so hiding it causes zero layout shift
- **Visible (MP playing):** `.hidden` removed by `updateTimerUI()`; bar shrinks from right to left as time passes
- **Color:** always red (`#e53935`), `.timer-warning` and `.timer-danger` classes are still toggled by JS but all map to the same red
- **Text labels** (`#timer-label`, `#timer-count`) kept in DOM for JS compatibility but hidden via CSS (`display: none`)
- **`#status-bar`** has `position: relative` so the absolute timer anchors to it correctly

---

## AI (`ai.js`)

- **`AIState`** — `{ enabled, playerNumber, thinking }`
- `startVsComputer()` — starts game with AI as Player 2
- `scheduleAIMove()` — called after each move; if AI's turn, triggers `runAIMove()` after 400ms delay
- Neural network loaded from `public/model/`; falls back to minimax if NN unavailable
- `loadNNModel()` called in `index.html` after scripts load

---

## CSS Architecture (`style.css`)

- **CSS variables** defined in `:root` (dark theme) and `html[data-theme="light"]`
- **Responsive cell size:** `--cell-size: min(calc((100vw - 52px) / 11), 58px)` — scales on mobile, caps at 58px
- **Layout stack** (top to bottom): `#score-bar` → `#status-bar` (contains `#timer-bar`) → `#game-area` → `#legend`
- **`#overlay`** inside `#game-area` with `position:absolute` covers the board for phase overlays
- **`.hidden`** is NOT a global utility class — each element has its own `ID.hidden { display:none }` rule, **except `#timer-bar.hidden`** which uses `opacity:0; visibility:hidden` to preserve layout

### Important CSS IDs/classes

| Selector | Purpose |
|----------|---------|
| `#score-bar` | Header bar |
| `#status-bar` | Turn info + timer container, flex-column |
| `#timer-bar` | Timer row inside status bar |
| `#board` | 11×11 CSS grid |
| `#overlay` | Phase overlays, `position:absolute` over board |
| `.cell.desert/forest/oasis` | Board zone colors |
| `.cell.valid-slide` / `.valid-knight` | Yellow/purple move highlights |
| `.cell.last-move-from` / `.last-move-to` | Cyan/green last move trace |
| `.horse.player1` / `.player2` | Piece appearance |
| `.horse.selected` | Gold glow + 1.22× scale |
| `.overlay-panel` | Centered card for overlays |

---

## Theme

- Toggled via `data-theme` attribute on `<html>` (`dark` default, `light`)
- Persisted in `localStorage('theme')`
- Theme button: `#theme-btn` shows ☀️ (dark) or 🌙 (light)

---

## Analytics

PostHog events fired at key moments: `game_started`, `game_restarted`, `piece_selected`, `move_executed`, `game_over`, `rules_opened`.

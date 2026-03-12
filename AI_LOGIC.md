# AI Logic Documentation

This document explains the logic behind the AI implemented in `public/ai.js` for the game. The AI uses a combination of heuristic evaluation and a neural network (if weights are loaded) within a Minimax search framework with alpha-beta pruning and iterative deepening.

## 1. Overview

The AI's decision-making process is centered around the `getAIMove` function, which acts as the root of the search. It uses a **Minimax algorithm** to explore future game states, assuming both players play optimally according to the evaluation function.

*   **Search Algorithm:** Minimax with Alpha-Beta pruning.
*   **Search Enhancement:** Iterative deepening (searches progressively deeper depths until a time limit is reached).
*   **Evaluation:** A hybrid approach using a handcrafted heuristic function (`evaluate()`) or a trained Neural Network (`nnEvaluate()`), depending on configuration and availability.
*   **Move Ordering:** Generates moves in a specific order to maximize the efficiency of alpha-beta pruning.

## 2. State Representation

The game state is represented internally for the AI using a "snapshot" object, which is decoupled from the main game's rendering state to allow for fast, pure simulation of moves without side effects.

A snapshot contains:
*   `horses`: An array of simplified piece objects `{ id, owner, col, row }`.
*   `board`: A dictionary mapping board coordinates (e.g., `'6,5'`) to piece objects for fast O(1) lookups.
*   `currentPlayer`: The player whose turn it is in this specific state (1 or 2).
*   `lastMoveToCenter`: A boolean indicating if the *previous* move resulted in a piece landing on the center square (the winning condition).

## 3. Evaluation Functions

The evaluation function's goal is to assign a numerical score to a given board state. A higher positive score means the state is better for the AI (Player 2, playing as Black), while a lower negative score means it's better for the opponent.

### 3.1 Neural Network Evaluation (`nnEvaluate`)

If the `model/weights.json` file is present and successfully loaded, and `FORCE_HEURISTIC` is false, the AI uses a small Multi-Layer Perceptron (MLP).

1.  **Feature Extraction (`stateToFeatures`):** The board state is flattened into an array of 41 floats:
    *   20 values for Player 1's piece coordinates (10 pieces * 2 coordinates: x, y).
    *   20 values for Player 2's piece coordinates.
    *   1 value indicating whose turn it is (0.0 for Player 1, 1.0 for Player 2).
    *   *Crucially, coordinates (1-11) are normalized to [0, 1] by subtracting 1 and dividing by 10.*
    *   Pieces are sorted consistently (by column, then row) before extraction to maintain consistent input structure.

2.  **Network Architecture:**
    *   Input Layer: 41 neurons.
    *   Hidden Layer 1: 128 neurons, ReLU activation.
    *   Hidden Layer 2: 64 neurons, ReLU activation.
    *   Output Layer: 1 neuron, Tanh activation (output range [-1, 1]).

3.  **Scaling and Orientation:** The output in [-1, 1] indicates how good the state is for the *current player to move*. Minimax always maximizes for `AIState.playerNumber` (the AI), so the score is oriented (negated if it's the opponent's turn) and scaled up (e.g., multiplied by 10000) to match the magnitude expected by the minimax search and move ordering priorities.

### 3.2 Heuristic Evaluation (`evaluate`)

If the neural net is not used, a handcrafted heuristic function evaluates the board based on several strategic principles. This function was completely rewritten in the latest update.

The total `score` is accumulated based on the following factors (positive favors AI, negative favors Opponent):

1.  **Terminal Threat Scoring (Highest Priority):**
    *   The game is won by landing on the center square `(6,6)`. A piece can only slide there if it is aligned on the central axis (column 6 or row 6) *and* has a friendly piece serving as a "backstop" on the opposite side of the center to stop the slide.
    *   **Ready Attack:** An aligned piece has a clear path to the center AND the required backstop cell is occupied by a friendly piece. This is a massive score boost (e.g., +3000 for AI, -4000 for Opponent).
    *   **Near Attack:** An aligned piece has a clear path, but the backstop is empty. Modest score boost (e.g., +400, -600).

2.  **Piece Development & Positioning:**
    *   **Proximity to Center:** Pieces are rewarded for being closer to the center (Manhattan distance).
    *   **Axis Alignment:** A large bonus for being on the central column (6) or row (6), as this is essential for attacking.
    *   **Corner Penalty:** Pieces stuck in their starting corners (e.g., col<=3, row<=3) are heavily penalized.
    *   **Backstop Occupation:** A bonus for occupying the four key backstop cells `(6,5)`, `(6,7)`, `(5,6)`, `(7,6)`, even if an attack isn't ready yet.

3.  **Mobility:**
    *   The number of valid moves available to all pieces. More options = better position. (AI mobility - Opponent mobility) * weight.

4.  **Piece Spread / Development Diversity:**
    *   Rewards moving pieces out of their starting corners. Encourages developing multiple pieces rather than just moving one piece repeatedly.

5.  **Axis Control:**
    *   Simply counts the number of pieces on the central column or row. (AI on axis - Opponent on axis) * weight.

6.  **Blocking Awareness:**
    *   If the opponent has a "Near Attack" (aligned, clear path, but no backstop), the AI checks if it has any pieces that can move into the path to block the potential slide. If it *cannot* block, a heavy penalty is applied.

## 4. Move Generation and Ordering

### 4.1 Pure Move Generators

The AI uses pure functions (no side effects on the global game state) to generate valid moves for a given snapshot:
*   `pureSlideMoves`: Calculates sliding moves along rows and columns until hitting an obstacle or the board edge.
*   `pureKnightMoves`: Calculates standard L-shaped knight jumps, restricted to landing only in the "DESERT" zone (outer ring) and only on empty squares.

### 4.2 Move Ordering (`generateOrderedMoves`)

To make Alpha-Beta pruning efficient, it is crucial to evaluate the "best" moves first. The `generateOrderedMoves` function assigns a priority to every possible move and sorts them descending.

Priority hierarchy:
1.  **Immediate Win:** Landing directly on `(6,6)`.
2.  **Creating a Backstop:** Moving a piece into a backstop cell when a friendly piece is already aligned and ready to attack.
3.  **Executing an Attack:** Moving an aligned piece when a friendly backstop is already in place.
4.  **Blocking:** Moving a piece into the path of an opponent's aligned piece.
5.  **Moving to Axis:** Moving a piece onto column 6 or row 6.
6.  **Proximity:** Fallback priority based on getting closer to the center.

## 5. Search Execution (Minimax)

The `getAIMove` function orchestrates the search:

1.  **Immediate Win Check:** If a move instantly wins the game, it is taken immediately without further search.
2.  **Iterative Deepening:** The search runs in a loop, starting at depth 3 and going up to depth 6.
    *   It uses a `timeBudget` (e.g., 12 seconds).
    *   A `deadline` is calculated. Inside the `minimax` recursive function, a node counter is checked periodically. If the deadline is passed, the search aborts early, returning the current alpha/beta bounds.
    *   The best move found at the deepest *fully completed* depth is chosen.
3.  **Alpha-Beta Pruning:** Inside `minimax`, `alpha` represents the minimum score the maximizing player (AI) is assured of, and `beta` represents the maximum score the minimizing player (Opponent) is assured of. If at any point the current score makes a branch worse than a previously explored option (e.g., `alpha >= beta`), that branch is "pruned" (skipped).

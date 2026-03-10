# 🐿️ Squirrel Oasis

A two-player browser strategy game. Race your squirrels to the acorn in the center of the board.

## How to Play

**Goal:** Be the first to move any squirrel onto the 🌰 acorn in the center.

**Setup:** White occupies the top-left and bottom-right corners. Black takes the top-right and bottom-left corners (5 squirrels each).

**On your turn, move one squirrel using one of two move types:**

- **Slide** — Move in a straight line (up/down/left/right). The squirrel slides until it hits the board edge or another squirrel, landing on the last empty cell.
- **Knight jump** — Jump in an L-shape (like a chess knight), but only to *desert* cells (the outer area of the board).

You cannot land on or jump over another squirrel.

## Board Zones

| Zone | Description |
|------|-------------|
| Desert | Outer area — knight jumps allowed here |
| Forest | Middle ring — slide only |
| Oasis | Center cell — land here to win |

## Play Online

Hosted on Firebase — open `index.html` or deploy with `firebase deploy`.

## Tech Stack

- Vanilla JS, HTML, CSS — no dependencies
- [PostHog](https://posthog.com) for analytics
- Firebase Hosting

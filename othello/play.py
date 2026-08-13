"""Play Othello against a trained agent from the terminal.

Usage:
    python -m othello.play --model othello_model.npz --human-color black
"""
import argparse

import numpy as np

from . import board as B
from .agent import FEATURE_DIM, choose_move
from .network import ValueNetwork

COLS = "abcdefgh"


def print_board(state):
    print("  " + " ".join(COLS))
    for r in range(B.SIZE):
        row = []
        for c in range(B.SIZE):
            v = state[r][c]
            row.append("." if v == B.EMPTY else ("B" if v == B.BLACK else "W"))
        print(f"{r + 1} " + " ".join(row))


def parse_move(text):
    text = text.strip().lower()
    if len(text) != 2 or text[0] not in COLS or not text[1].isdigit():
        return None
    c = COLS.index(text[0])
    r = int(text[1]) - 1
    if not (0 <= r < B.SIZE):
        return None
    return (r, c)


def main():
    parser = argparse.ArgumentParser(description="Play Othello against the trained agent")
    parser.add_argument("--model", type=str, default="othello_model.npz")
    parser.add_argument("--human-color", choices=["black", "white"], default="black")
    args = parser.parse_args()

    net = ValueNetwork(FEATURE_DIM)
    net.load(args.model)
    rng = np.random.default_rng()

    human = B.BLACK if args.human_color == "black" else B.WHITE
    state = B.initial_board()
    player = B.BLACK

    while not B.is_game_over(state):
        moves = B.legal_moves(state, player)
        print()
        print_board(state)
        black, white = B.score(state)
        print(f"Black: {black}  White: {white}")

        if not moves:
            print(f"{'Black' if player == B.BLACK else 'White'} has no legal move, passing.")
            player = -player
            continue

        if player == human:
            move = None
            while move not in moves:
                text = input(f"Your move ({'Black' if human == B.BLACK else 'White'}), e.g. d3: ")
                move = parse_move(text)
                if move not in moves:
                    print("Illegal move, try again. Legal moves: " +
                          ", ".join(f"{COLS[c]}{r + 1}" for r, c in moves))
            state = B.apply_move(state, move, player)
        else:
            move = choose_move(state, player, net, epsilon=0.0, rng=rng)
            print(f"AI plays {COLS[move[1]]}{move[0] + 1}")
            state = B.apply_move(state, move, player)
        player = -player

    print()
    print_board(state)
    black, white = B.score(state)
    print(f"Final score - Black: {black}  White: {white}")
    result = B.winner(state)
    if result == B.EMPTY:
        print("Draw!")
    else:
        print("Black wins!" if result == B.BLACK else "White wins!")


if __name__ == "__main__":
    main()

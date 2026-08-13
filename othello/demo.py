"""Watch the trained agent play itself in the terminal (no human input).

Usage:
    python -m othello.demo --model othello_model.npz --delay 0.6 --games 3
"""
import argparse
import time

import numpy as np

from . import board as B
from .agent import FEATURE_DIM, choose_move
from .network import ValueNetwork
from .play import print_board

COLS = "abcdefgh"


def play_one_game(net, delay, epsilon, rng):
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

        move = choose_move(state, player, net, epsilon=epsilon, rng=rng)
        print(f"{'Black' if player == B.BLACK else 'White'} plays {COLS[move[1]]}{move[0] + 1}")
        state = B.apply_move(state, move, player)
        player = -player
        time.sleep(delay)

    print()
    print_board(state)
    black, white = B.score(state)
    print(f"Final score - Black: {black}  White: {white}")
    result = B.winner(state)
    if result == B.EMPTY:
        print("Draw!")
    else:
        print("Black wins!" if result == B.BLACK else "White wins!")
    return result


def main():
    parser = argparse.ArgumentParser(description="Watch the trained agent play itself")
    parser.add_argument("--model", type=str, default="othello_model.npz")
    parser.add_argument("--delay", type=float, default=0.6,
                         help="seconds to pause between moves")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.1,
                         help="chance of a random move, so repeated demos aren't identical")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    net = ValueNetwork(FEATURE_DIM)
    net.load(args.model)
    rng = np.random.default_rng(args.seed)

    for g in range(1, args.games + 1):
        print(f"\n===== Game {g}/{args.games} =====")
        play_one_game(net, args.delay, args.epsilon, rng)


if __name__ == "__main__":
    main()

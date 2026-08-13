"""Play a few full games with the trained agent and dump them as JSON so a
static web page can replay them move-by-move without needing to run Python
or re-implement the game rules in JavaScript.

Usage:
    python -m othello.export_demo_games --model othello_model.npz --games 5 --out demo_games.json --html demo.html
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import board as B
from .agent import FEATURE_DIM, choose_move
from .network import ValueNetwork

COLS = "abcdefgh"


def board_to_list(state):
    return [[int(v) for v in row] for row in state]


def square_name(move):
    if move is None:
        return None
    r, c = move
    return f"{COLS[c]}{r + 1}"


def play_one_game(net, epsilon, rng):
    state = B.initial_board()
    player = B.BLACK
    black, white = B.score(state)
    steps = [{
        "board": board_to_list(state),
        "mover": None,
        "move": None,
        "passed": False,
        "black": black,
        "white": white,
    }]

    while not B.is_game_over(state):
        moves = B.legal_moves(state, player)
        mover_name = "black" if player == B.BLACK else "white"

        if not moves:
            player = -player
            black, white = B.score(state)
            steps.append({
                "board": board_to_list(state),
                "mover": mover_name,
                "move": None,
                "passed": True,
                "black": black,
                "white": white,
            })
            continue

        move = choose_move(state, player, net, epsilon=epsilon, rng=rng)
        state = B.apply_move(state, move, player)
        black, white = B.score(state)
        steps.append({
            "board": board_to_list(state),
            "mover": mover_name,
            "move": square_name(move),
            "passed": False,
            "black": black,
            "white": white,
        })
        player = -player

    result = B.winner(state)
    result_name = "draw" if result == B.EMPTY else ("black" if result == B.BLACK else "white")
    return {"result": result_name, "steps": steps}


def main():
    parser = argparse.ArgumentParser(description="Export self-play games as JSON for a web demo")
    parser.add_argument("--model", type=str, default="othello_model.npz")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.1,
                         help="chance of a random move, so games differ from each other")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="demo_games.json")
    parser.add_argument("--html", type=str, default=None,
                         help="also write a standalone HTML viewer to this path "
                              "(embeds the game data, no server needed)")
    args = parser.parse_args()

    net = ValueNetwork(FEATURE_DIM)
    net.load(args.model)
    rng = np.random.default_rng(args.seed)

    games = [play_one_game(net, args.epsilon, rng) for _ in range(args.games)]
    games_json = json.dumps({"games": games}, ensure_ascii=False)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(games_json)
    print(f"Wrote {len(games)} games ({sum(len(g['steps']) for g in games)} total steps) to {args.out}")

    if args.html:
        template_path = Path(__file__).parent / "webdemo_template.html"
        template = template_path.read_text(encoding="utf-8")
        page = template.replace("__GAMES_DATA_JSON__", games_json)
        Path(args.html).write_text(page, encoding="utf-8")
        print(f"Wrote standalone viewer to {args.html} (open it directly in a browser)")


if __name__ == "__main__":
    main()

"""Feature encoding, move selection, and one self-play game.

Learning approach: the network estimates V(board, player_to_move) ~= the
eventual game outcome (+1 win / -1 loss / 0 draw) for the player about to
move. During self-play we pick moves greedily (epsilon-greedy) against this
estimate, then after the game finishes we know the true outcome and train
the network towards it for every position that was visited (Monte-Carlo /
self-play outcome regression -- a simplified relative of TD learning).
"""
import numpy as np

from . import board as B

CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]
FEATURE_DIM = B.SIZE * B.SIZE + 4  # 64 board cells + 4 hand-crafted features


def encode(board_state, player):
    """Encode a board from the perspective of `player` (about to move)."""
    own = (board_state == player).astype(np.float64)
    opp = (board_state == -player).astype(np.float64)
    cells = (own - opp).flatten()

    my_moves = len(B.legal_moves(board_state, player))
    opp_moves = len(B.legal_moves(board_state, -player))
    my_corners = sum(1 for r, c in CORNERS if board_state[r][c] == player)
    opp_corners = sum(1 for r, c in CORNERS if board_state[r][c] == -player)

    extra = np.array([my_moves / 20.0, opp_moves / 20.0,
                       my_corners / 4.0, opp_corners / 4.0])
    return np.concatenate([cells, extra])


def choose_move(board_state, player, net, epsilon, rng):
    """Pick a move for `player`. With prob. epsilon, pick a random legal move;
    otherwise evaluate every resulting position and take the best one."""
    moves = B.legal_moves(board_state, player)
    if not moves:
        return None
    if rng.random() < epsilon:
        return moves[rng.integers(len(moves))]

    best_move, best_value = None, None
    for move in moves:
        next_board = B.apply_move(board_state, move, player)
        feat = encode(next_board, -player)
        value_for_opponent = net.predict(feat)[0]
        value_for_me = -value_for_opponent
        if best_value is None or value_for_me > best_value:
            best_value, best_move = value_for_me, move
    return best_move


def play_self_play_game(net, epsilon, rng):
    """Play one game of the network against itself.

    Returns (features, targets, winner) where features/targets are arrays
    ready to feed into ValueNetwork.train_step.
    """
    state = B.initial_board()
    player = B.BLACK
    records = []  # (feature vector, player color at that position)

    while not B.is_game_over(state):
        moves = B.legal_moves(state, player)
        if not moves:
            player = -player
            continue
        records.append((encode(state, player), player))
        move = choose_move(state, player, net, epsilon, rng)
        state = B.apply_move(state, move, player)
        player = -player

    result = B.winner(state)
    if not records:
        return np.empty((0, FEATURE_DIM)), np.empty((0,)), result

    features = np.array([f for f, _ in records])
    targets = np.array([
        0.0 if result == B.EMPTY else (1.0 if result == p else -1.0)
        for _, p in records
    ])
    return features, targets, result

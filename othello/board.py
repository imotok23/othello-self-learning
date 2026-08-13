"""Othello board representation and rules (no AI logic here)."""
import numpy as np

EMPTY, BLACK, WHITE = 0, 1, -1
SIZE = 8
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def initial_board():
    b = np.zeros((SIZE, SIZE), dtype=np.int8)
    b[3][3] = WHITE
    b[3][4] = BLACK
    b[4][3] = BLACK
    b[4][4] = WHITE
    return b


def in_bounds(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE


def _flips_for_move(board, r, c, player):
    if board[r][c] != EMPTY:
        return []
    opponent = -player
    flips = []
    for dr, dc in DIRECTIONS:
        line = []
        rr, cc = r + dr, c + dc
        while in_bounds(rr, cc) and board[rr][cc] == opponent:
            line.append((rr, cc))
            rr += dr
            cc += dc
        if line and in_bounds(rr, cc) and board[rr][cc] == player:
            flips.extend(line)
    return flips


def legal_moves(board, player):
    moves = []
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] == EMPTY and _flips_for_move(board, r, c, player):
                moves.append((r, c))
    return moves


def apply_move(board, move, player):
    r, c = move
    flips = _flips_for_move(board, r, c, player)
    if not flips:
        raise ValueError(f"Illegal move {move} for player {player}")
    new_board = board.copy()
    new_board[r][c] = player
    for fr, fc in flips:
        new_board[fr][fc] = player
    return new_board


def is_game_over(board):
    return not legal_moves(board, BLACK) and not legal_moves(board, WHITE)


def score(board):
    black = int(np.sum(board == BLACK))
    white = int(np.sum(board == WHITE))
    return black, white


def winner(board):
    black, white = score(board)
    if black > white:
        return BLACK
    if white > black:
        return WHITE
    return EMPTY

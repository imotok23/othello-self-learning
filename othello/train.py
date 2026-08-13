"""Train the self-play Othello agent.

Usage:
    python -m othello.train --episodes 20000 --out othello_model.npz
"""
import argparse
import sys
import time

import numpy as np

from . import board as B
from .agent import FEATURE_DIM, choose_move, play_self_play_game
from .network import ValueNetwork


def format_progress_bar(frac, elapsed, total_episodes, done_episodes, width=20):
    filled = int(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    eta = elapsed / done_episodes * (total_episodes - done_episodes) if done_episodes else 0.0
    return (f"\r[{bar}] {frac * 100:5.1f}%  "
            f"{done_episodes}/{total_episodes}episode  "
            f"経過{elapsed:.0f}s  残り約{eta:.0f}s   ")


class ReplayBuffer:
    """Fixed-size circular buffer of (feature, target) pairs for minibatch
    training, decoupled from the single game that produced them."""

    def __init__(self, capacity, feature_dim, rng):
        self.capacity = capacity
        self.features = np.empty((capacity, feature_dim))
        self.targets = np.empty(capacity)
        self.size = 0
        self.pos = 0
        self.rng = rng

    def add(self, features, targets):
        n = len(targets)
        if n == 0:
            return
        if n > self.capacity:
            features, targets = features[-self.capacity:], targets[-self.capacity:]
            n = self.capacity
        end = self.pos + n
        if end <= self.capacity:
            self.features[self.pos:end] = features
            self.targets[self.pos:end] = targets
        else:
            first = self.capacity - self.pos
            self.features[self.pos:] = features[:first]
            self.targets[self.pos:] = targets[:first]
            self.features[:end - self.capacity] = features[first:]
            self.targets[:end - self.capacity] = targets[first:]
        self.pos = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=min(batch_size, self.size))
        return self.features[idx], self.targets[idx]


def random_move(state, player, rng):
    moves = B.legal_moves(state, player)
    return moves[rng.integers(len(moves))] if moves else None


def play_eval_game(net, net_color, rng):
    state = B.initial_board()
    player = B.BLACK
    while not B.is_game_over(state):
        moves = B.legal_moves(state, player)
        if not moves:
            player = -player
            continue
        if player == net_color:
            move = choose_move(state, player, net, epsilon=0.0, rng=rng)
        else:
            move = random_move(state, player, rng)
        state = B.apply_move(state, move, player)
        player = -player
    return B.winner(state)


def evaluate(net, games, rng):
    wins = draws = losses = 0
    for i in range(games):
        net_color = B.BLACK if i % 2 == 0 else B.WHITE
        result = play_eval_game(net, net_color, rng)
        if result == net_color:
            wins += 1
        elif result == B.EMPTY:
            draws += 1
        else:
            losses += 1
    return wins, draws, losses


def main():
    parser = argparse.ArgumentParser(description="Train a self-play Othello agent")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="othello_model.npz")
    parser.add_argument("--epsilon-start", type=float, default=0.3)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--no-augment", action="store_true",
                         help="盤面の回転・鏡映による8倍データ拡張を無効化する")
    parser.add_argument("--replay-size", type=int, default=20000,
                         help="経験リプレイバッファの容量（局面数）")
    parser.add_argument("--batch-size", type=int, default=256,
                         help="リプレイバッファからのミニバッチサイズ")
    parser.add_argument("--train-steps", type=int, default=4,
                         help="自己対戦1局あたりのミニバッチ学習回数")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    net = ValueNetwork(FEATURE_DIM, hidden_dim=args.hidden, lr=args.lr, seed=args.seed)
    buffer = ReplayBuffer(args.replay_size, FEATURE_DIM, rng)

    start = time.time()
    last_bar_write = 0.0
    for ep in range(1, args.episodes + 1):
        frac = ep / args.episodes
        epsilon = args.epsilon_start + (args.epsilon_end - args.epsilon_start) * frac
        feats, targets, _ = play_self_play_game(net, epsilon, rng, augment=not args.no_augment)
        buffer.add(feats, targets)

        if buffer.size >= args.batch_size:
            for _ in range(args.train_steps):
                batch_feats, batch_targets = buffer.sample(args.batch_size)
                net.train_step(batch_feats, batch_targets)

        elapsed = time.time() - start
        # 進捗バーの書き換えは0.1秒おきに間引いて、出力自体が学習を遅らせないようにする
        if elapsed - last_bar_write >= 0.1 or ep == args.episodes:
            sys.stdout.write(format_progress_bar(frac, elapsed, args.episodes, ep))
            sys.stdout.flush()
            last_bar_write = elapsed

        if ep % args.eval_every == 0:
            wins, draws, losses = evaluate(net, args.eval_games, rng)
            elapsed = time.time() - start
            win_rate = wins / args.eval_games
            sys.stdout.write("\n")
            print(f"episode {ep:6d} | eps={epsilon:.3f} | vs random: "
                  f"{wins}W/{draws}D/{losses}L ({win_rate:.0%}) | {elapsed:.0f}s")
            net.save(args.out)

    sys.stdout.write("\n")
    net.save(args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()

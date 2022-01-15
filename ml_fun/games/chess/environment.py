from typing import Tuple

import chess
import gym

Observation = str

# Ideas for board representation: https://github.com/crypt3lx2k/Zerofish/blob/e2923479c26ec92fe34046f0ec51f90f838c8d13/adapter.py#L103
# See chess.Move for move representation: https://python-chess.readthedocs.io/en/latest/_modules/chess.html#Move


class ChessEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board = chess.Board()

    def reset(self) -> Observation:
        self.board = chess.Board()
        return self.board.fen()

    def render(self, mode: str = "human") -> None:
        print(f"FEN: {self.board.fen()}")
        print(self.board)

    def step(self, action: str) -> Tuple[Observation, int, bool, None]:
        move = chess.Move.from_uci(action)
        if not self.board.is_legal(move):
            done = True
            reward = -1 if self.board.turn else 1
            observation = self.board.fen()
        else:
            self.board.push(move)
            outcome = self.board.outcome()
            if outcome is None:
                done = False
                reward = 0
            else:
                done = True
                reward = 1 if outcome.winner else -1
            observation = self.board.fen()

        return observation, reward, done, None

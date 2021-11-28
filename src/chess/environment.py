from typing import Tuple

import gym

import chess

Observation = str


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

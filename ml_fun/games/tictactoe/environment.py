from copy import deepcopy
from typing import List, Tuple

import gym
import numpy as np

Observation = Tuple[np.ndarray, bool]


class TicTacToeEnv(gym.Env):
    def __init__(self, show_number: bool = False, board_size: int = 3):
        super().__init__()
        self.show_number = show_number
        self.board_size = board_size
        self.board = np.zeros(shape=(self.board_size, self.board_size), dtype="int8")
        self.action_space = gym.spaces.Discrete(len(self.board))
        self.player_ones_turn: bool = True
        self.done = False
        self.marks = {0: "0", 1: "X", -1: "O"}
        self.reset()

    def reset(self) -> Observation:
        return self.set_state(
            board=np.zeros(shape=(self.board_size, self.board_size), dtype="int8"),
            player_ones_turn=True,
        )

    def set_state(self, board: np.ndarray, player_ones_turn: bool) -> Observation:
        assert (
            self.board.shape == board.shape
        ), f"Want: {self.board.shape} Got: {board.shape}"
        assert (
            self.board.dtype == board.dtype
        ), f"Want: {self.board.dtype} Got: {board.dtype}"
        self.board = deepcopy(board)
        self.player_ones_turn = player_ones_turn

        return self._get_observation()

    def _get_observation(self) -> Observation:
        return deepcopy(self.board), self.player_ones_turn

    def render(self, mode: str = "human") -> None:
        # print(f"Next to act: {self.marks[1 if self.player_ones_turn else -1]}")
        for y in range(self.board_size):
            for x in range(self.board_size):
                state = self.board[y, x]
                if state == 0 and self.show_number:
                    mark = str(y * self.board_size + x)
                else:
                    mark = self.marks[state]
                print(mark, end="|" if x + 1 < self.board_size else "\n")
            print("-" * (2 * self.board_size - 1) if y + 1 < self.board_size else "")

    def available_actions(self) -> List[int]:
        actions = []
        for action in range(self.board_size * self.board_size):
            y = action // self.board_size
            x = action % self.board_size
            if self.board[y, x] == 0:
                actions.append(action)

        return actions

    def step(self, action: int) -> Tuple[Observation, int, bool, None]:
        assert (
            0 <= action < self.board_size * self.board_size
        ), f"Encountered illegal action `{action}` of type `{type(action)}`"

        y = action // self.board_size
        x = action % self.board_size

        if self.board[y, x] == 0:
            self.board[y, x] = 1 if self.player_ones_turn else -1
            done, reward = self._check_game_state()
        else:
            done = True
            reward = -1 if self.player_ones_turn else 1

        self.player_ones_turn = not self.player_ones_turn

        return self._get_observation(), reward, done, None

    def _check_game_state(self) -> Tuple[bool, int]:
        # check columns
        for player in [1, -1]:
            for y in range(self.board_size):
                is_line: bool = True
                if not is_line:
                    break
                for x in range(self.board_size):
                    if self.board[y, x] != player:
                        is_line = False
                        break
                if is_line:
                    return True, player

        # check rows
        for player in [1, -1]:
            for x in range(self.board_size):
                is_line: bool = True
                if not is_line:
                    break
                for y in range(self.board_size):
                    if self.board[y, x] != player:
                        is_line = False
                        break
                if is_line:
                    return True, player

        # check diagonal /
        for player in [1, -1]:
            is_line: bool = True
            for offset in range(self.board_size):
                x = self.board.shape[0] - offset - 1
                y = offset
                if self.board[y, x] != player:
                    is_line = False
                    break
            if is_line:
                return True, player

        # check diagonal \
        for player in [1, -1]:
            is_line: bool = True
            for offset in range(self.board_size):
                x = offset
                y = offset
                if self.board[y, x] != player:
                    is_line = False
                    break
            if is_line:
                return True, player

        # potential draw
        return (
            np.count_nonzero(self.board) == self.board.shape[0] * self.board.shape[1],
            0,
        )

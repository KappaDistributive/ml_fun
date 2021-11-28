import numpy as np
import pytest

from tictactoe.environment import TicTacToeEnv


@pytest.fixture
def environment(board_size: int = 3) -> TicTacToeEnv:
    env = TicTacToeEnv(board_size=board_size)
    env.reset()
    return env


def test_winning_condition(environment: TicTacToeEnv) -> None:
    setups = [
        (
            np.array([[-1, 0, -1], [1, 0, 1], [0, 0, 0]], dtype="int8"),
            True,
            4,
            np.array([[-1, 0, -1], [1, 1, 1], [0, 0, 0]], dtype="int8"),
            1,
            True,
        ),
        (
            np.array([[-1, 0, -1], [1, 0, 1], [0, 1, 0]], dtype="int8"),
            False,
            1,
            np.array([[-1, -1, -1], [1, 0, 1], [0, 1, 0]], dtype="int8"),
            -1,
            True,
        ),
        (
            np.array([[1, 0, -1], [0, 1, -1], [0, 0, 0]], dtype="int8"),
            True,
            8,
            np.array([[1, 0, -1], [0, 1, -1], [0, 0, 1]], dtype="int8"),
            1,
            True,
        ),
        (
            np.array([[1, 0, -1], [0, -1, 1], [0, 1, 0]], dtype="int8"),
            False,
            6,
            np.array([[1, 0, -1], [0, -1, 1], [-1, 1, 0]], dtype="int8"),
            -1,
            True,
        ),
    ]

    for setup in setups:
        board, player_ones_turn, action, want_result, want_reward, want_done = setup
        environment.set_state(board, player_ones_turn)
        got_observation, got_reward, got_done, _ = environment.step(action)
        np.testing.assert_array_equal(got_observation[0], want_result)
        assert got_done == want_done
        assert got_reward == want_reward

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from src.games.tictactoe.environment import TicTacToeEnv

# work-around for missing type-annotion
if TYPE_CHECKING:
    from pytest import FixtureRequest as __FixtureRequest

    class FixtureRequest(__FixtureRequest):
        param: Any


else:
    from pytest import FixtureRequest


@pytest.fixture(scope="function")
def environment(request: FixtureRequest) -> TicTacToeEnv:
    print(type(request))
    env = TicTacToeEnv(board_size=request.param)
    env.reset()
    return env


@pytest.mark.parametrize("environment", (2,), indirect=True)
def test_winning_condition_small(environment: TicTacToeEnv) -> None:
    setups = [
        (
            np.array([[1, 0], [-1, 0]], dtype="int8"),
            True,
            3,
            np.array([[1, 0], [-1, 1]], dtype="int8"),
            1,
            True,
        ),
        (
            np.array([[1, 0], [-1, 0]], dtype="int8"),
            True,
            1,
            np.array([[1, 1], [-1, 0]], dtype="int8"),
            1,
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


@pytest.mark.parametrize("environment", (3,), indirect=True)
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

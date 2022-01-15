"""
Play Tic Tac Toe via MCTS
"""
import random
from copy import deepcopy
from typing import Tuple

from tqdm import tqdm

from src.games.tictactoe.environment import TicTacToeEnv
from src.mcts.mcts import MCTS, Action, Node, State


class TicTacToeMCTS(MCTS):
    def __init__(self, initial_state: State, initial_player: int):
        super().__init__(initial_state=initial_state, initial_player=initial_player)

    def _random_child(self, node: Node) -> Tuple[Action, Node]:
        action = random.choice(range(9))
        if action not in node.children:
            env = TicTacToeEnv()
            env.reset()
            env.board = deepcopy(self.initial_state)
            env.player_ones_turn = node.to_play == 1
            observation, reward, done, _ = env.step(action)
            node.children[action] = Node(
                to_play=-1 * node.to_play,
                state=deepcopy(observation[0]),
                reward=reward,
                is_terminal=done,
            )

        return action, node.children[action]

    def _add_children(self, node: Node) -> None:
        for action in range(9):
            if action not in node.children:
                env = TicTacToeEnv()
                env.reset()
                env.board = deepcopy(node.state)
                env.player_ones_turn = node.to_play == 1
                observation, reward, done, _ = env.step(action)
                node.children[action] = Node(
                    to_play=-1 * node.to_play,
                    state=deepcopy(observation[0]),
                    reward=reward,
                    is_terminal=done,
                )

    def _simulate(self, node: Node) -> float:
        env = TicTacToeEnv()
        env.reset()
        env.board = deepcopy(node.state)
        env.player_ones_turn = node.to_play == 1

        done, reward = env._check_game_state()
        while not done:
            action = random.choice(env.available_actions())
            observation, reward, done, _ = env.step(action)

        return reward


if __name__ == "__main__":
    # TODO: reimplement this via a dummy MuZero agent.
    # TODO: train MuZero agent.
    env = TicTacToeEnv(show_number=True)
    env.reset()

    player = 2 * random.randint(0, 1) - 1

    done = False
    reward = 0

    print(f"\nAgent is {'O' if player == 1 else 'X'}")
    env.render()
    while not done:
        print(f"Player to move: {'X' if env.player_ones_turn else 'O'}")
        if player == 1:
            action = int(input("Enter your move:"))
            # action = random.choice(env.available_actions())
        else:
            mcts = TicTacToeMCTS(deepcopy(env.board), initial_player=player)
            for _ in tqdm(range(1_000)):
                mcts.rollout(mcts.root)

            # available_actions = env.available_actions()
            max_value = max(
                child.value()
                for action, child in mcts.root.children.items()
                if action in env.available_actions()
            )
            print(
                {
                    action: (child.value())
                    for action, child in mcts.root.children.items()
                }
            )
            action, _ = random.choice(
                [
                    (a, c)
                    for a, c in mcts.root.children.items()
                    if c.visit_count > 0
                    and c.value() == max_value
                    and a in env.available_actions()
                ]
            )

        print(f"Action: {action}")
        observation, reward, done, _ = env.step(action)
        env.render()
        player *= -1

    if reward == 0:
        print("Draw")
    elif reward == 1:
        print("X wins")
    else:
        print("O wins")

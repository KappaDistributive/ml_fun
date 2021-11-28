"""
Play Tic Tac Toe via MCTS
"""
import random
from copy import deepcopy
from typing import Tuple

from tqdm import tqdm

from src.tictactoe.environment import TicTacToeEnv
from mcts.mcts import Action, MCTS, Node, State


class TicTacToeMCTS(MCTS):
    def __init__(self, initial_state: State):
        super().__init__(initial_state=initial_state, initial_player=-1)

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
    env = TicTacToeEnv(show_number=True)
    env.reset()

    player = -1

    done = False
    reward = 0

    env.render()
    while not done:
        if player == 1:
            action = int(input("Enter your move:"))
        else:
            mcts = TicTacToeMCTS(deepcopy(env.board))
            for _ in tqdm(range(1_000)):
                mcts.rollout(mcts.root)
            max_value = max(child.value() for child in mcts.root.children.values())
            print(
                {action: child.value() for action, child in mcts.root.children.items()}
            )
            action, _ = random.choice(
                [
                    (a, c)
                    for a, c in mcts.root.children.items()
                    if c.visit_count > 0 and c.value() == max_value
                ]
            )

        print(f"Action: {action}")
        observation, reward, done, _ = env.step(action)
        env.render()
        player *= -1

    print(reward)

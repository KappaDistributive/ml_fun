from src.reinforcement_learning.environments.cartpole import CartPole
from src.reinforcement_learning.search.mcts.mcts import MCTS

if __name__ == "__main__":
    environment = CartPole()
    search = MCTS()

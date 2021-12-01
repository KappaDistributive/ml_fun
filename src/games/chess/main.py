from src.games.chess.environment import ChessEnv

if __name__ == "__main__":
    environment = ChessEnv()
    environment.reset()
    environment.render()

    done = False

    while not done:
        action = input("Enter your move:")
        _, reward, done, _ = environment.step(action)
        environment.render()

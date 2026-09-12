import numpy as np
from engine.game import Game
from engine.enums import GameState
from src.agents.good_agent import GoodAgent

def generate_dataset(num_games=1000, width=9, height=9, num_mines=10, filename="minesweeper_dataset.npz"):
    """
    Plays headless games using GoodAgent and records the board state and ground truth.
    """
    X_data = []  # The visible board (Input)
    Y_data = []  # The actual mine locations (Target)

    agent = GoodAgent()
    
    print(f"Generating dataset from {num_games} games...")
    
    for game_idx in range(num_games):
        game = Game(width, height, num_mines)
        
        while game.state == GameState.ONGOING:
            # The engine doesn't place mines until the first click to guarantee safety.
            # We skip saving the completely empty board.
            if game.first_click:
                x, y = agent.select_action(game.board.board, game.board.revealed, game.board.flags)
                game.click(x, y)
                continue

            # --- 1. Extract Visible State (X) ---
            # We initialize a grid of -1 (representing unrevealed cells)
            visible_state = np.full((height, width), -1, dtype=np.int8)
            
            # Map the revealed numbers (0-8) onto the grid
            for y in range(height):
                for x in range(width):
                    if (x, y) in game.board.revealed:
                        visible_state[y, x] = game.board.board[y][x]
            
            # --- 2. Extract Ground Truth (Y) ---
            # We initialize a grid of 0s (safe), and place 1s where the mines actually are
            ground_truth = np.zeros((height, width), dtype=np.int8)
            for (mx, my) in game.board.mines:
                ground_truth[my, mx] = 1
                
            # Save the snapshot
            X_data.append(visible_state)
            Y_data.append(ground_truth)
            
            # Advance the game
            x, y = agent.select_action(game.board.board, game.board.revealed, game.board.flags)
            game.click(x, y)
            
        if (game_idx + 1) % 500 == 0:
            print(f"Completed {game_idx + 1} / {num_games} games. Collected {len(X_data)} samples.")
            
    # Convert lists to NumPy arrays
    X_array = np.array(X_data)
    Y_array = np.array(Y_data)
    
    print("\nDataset Generation Complete!")
    print(f"X shape (Input):  {X_array.shape}")
    print(f"Y shape (Target): {Y_array.shape}")
    
    # Save as a highly compressed NumPy archive
    np.savez_compressed(filename, X=X_array, Y=Y_array)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    # 5,000 games will generate roughly 50,000 to 70,000 training examples
    generate_dataset(num_games=5000, width=9, height=9, num_mines=10)
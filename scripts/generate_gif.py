import sys
import copy
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

# Add the project root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.engine.game import Game
from src.engine.enums import GameState
from src.ml.model import MinesweeperCNN
from src.agents.neural_agent import NeuralAgent

def draw_board(ax, board):
    """Helper function to visually render the Minesweeper board."""
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    
    height, width = board.height, board.width
    
    # Draw a simple grid
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5) 
    
    for y in range(height):
        for x in range(width):
            is_revealed = (x, y) in board.revealed
            is_mine = (x, y) in board.mines
            
            # Fetch value directly from your 2D board array
            value = board.board[y][x]
            
            # Draw cell borders
            cell_color = 'white' if is_revealed else 'lightgray'
            rect = patches.Rectangle(
                (x - 0.5, y - 0.5), 1, 1, 
                fill=True, 
                edgecolor='gray', 
                facecolor=cell_color
            )
            ax.add_patch(rect)
            
            if is_revealed:
                if is_mine:
                    ax.text(x, y, '💣', ha='center', va='center', fontsize=14)
                elif value > 0:
                    colors = ['blue', 'green', 'red', 'purple', 'maroon', 'cyan', 'black', 'gray']
                    color = colors[value - 1] if 1 <= value <= 8 else 'black'
                    ax.text(x, y, str(value), ha='center', va='center', 
                            fontsize=12, fontweight='bold', color=color)

def generate_gameplay_gif():
    print("Initializing environment and loading model...")
    game = Game(width=9, height=9, num_mines=10)
    
    checkpoint_path = Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = checkpoint_path.parent / "model_50000_games.pt"
        
    agent = NeuralAgent(model_path=str(checkpoint_path), width=9, height=9)
    
    frames = []
    
    print("Bot is playing a game...")
    frames.append(copy.deepcopy(game.board))
    
    # Loop while the game state is strictly ONGOING
    while game.state == GameState.ONGOING:
        # Pass your Board attributes directly into the agent's expected parameters
        x, y = agent.select_action(game.board.board, game.board.revealed, game.board.flags)
        
        game.click(x, y)
        frames.append(copy.deepcopy(game.board))
        
    print(f"Game finished in {len(frames) - 1} moves! Win status: {game.state == GameState.WON}")
    print("Rendering GIF... (This might take a minute)")
    
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.tight_layout()
    
    def update(frame_idx):
        draw_board(ax, frames[frame_idx])
        ax.set_title(f"Neural Agent - Move {frame_idx}", fontsize=14)
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=500) 
    
    output_path = Path(__file__).resolve().parent.parent / "assets" / "demo.gif"
    anim.save(output_path, writer=PillowWriter(fps=2))
    
    print(f"Success! Saved gameplay GIF to {output_path}")

if __name__ == "__main__":
    generate_gameplay_gif()
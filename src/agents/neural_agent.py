import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseAgent
from src.ml.model import MinesweeperCNN

class NeuralAgent(BaseAgent):
    """
    An agent that uses a trained PyTorch Convolutional Neural Network 
    to evaluate the safest move on the board.
    """
    
    def __init__(self, model_path, width=9, height=9):
        """
        Loads the trained weights into the CNN and sets it to evaluation mode.
        """
        self.width = width
        self.height = height
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        
        # Initialize model and load weights
        self.model = MinesweeperCNN().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        # VERY IMPORTANT: Set model to evaluation mode (disables dropouts/gradients)
        self.model.eval()

    def select_action(self, board, revealed_set, flags_set):
        # FIX: Dynamically read board size instead of using hardcoded self.height/width
        height = len(board)
        width = len(board[0])

        grid = np.full((height, width), -1, dtype=np.int8)
        for x, y in revealed_set:
            grid[y, x] = board[y][x]

        # Tensor Translation
        grid_shifted = np.where(grid == -1, 9, grid)
        x_tensor = torch.tensor(grid_shifted, dtype=torch.long)
        x_one_hot = F.one_hot(x_tensor, num_classes=10)
        x_one_hot = x_one_hot.permute(2, 0, 1).to(torch.float32).unsqueeze(0).to(self.device)

        # The Forward Pass
        with torch.no_grad():
            probabilities = self.model(x_one_hot).squeeze()
            
        prob_map = probabilities.numpy()

        # Vectorized Action Masking
        for x, y in revealed_set:
            prob_map[y, x] = 1.0
        for x, y in flags_set:
            prob_map[y, x] = 1.0

        # Find Safest Coordinate
        flat_idx = np.argmin(prob_map)
        best_y, best_x = np.unravel_index(flat_idx, prob_map.shape)

        return (int(best_x), int(best_y))
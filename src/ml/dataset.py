import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class MinesweeperDataset(Dataset):
    """
    Loads the saved NumPy dataset and applies one-hot encoding, 
    formatting the tensors for a Convolutional Neural Network.
    """
    def __init__(self, npz_path):
        """
        Args:
            npz_path (str): The file path to the generated .npz dataset.
        """
        # Load the compressed data into memory
        data = np.load(npz_path)
        self.X_raw = data['X']
        self.Y_raw = data['Y']
        
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.X_raw)
        
    def __getitem__(self, idx):
        """
        Retrieves a single game state and its answer key, transforming 
        them into PyTorch tensors.
        """
        x_grid = self.X_raw[idx]
        y_grid = self.Y_raw[idx]
        
        # --- X (Input) Transformation ---
        # Replace -1 (hidden cells) with 9 so all values are >= 0.
        # This gives us 10 categorical classes (0-8 for numbers, 9 for hidden).
        x_grid_shifted = np.where(x_grid == -1, 9, x_grid)
        
        # Convert to a PyTorch tensor of 64-bit integers (required for one_hot)
        x_tensor = torch.tensor(x_grid_shifted, dtype=torch.long)
        
        # Apply one-hot encoding. 
        # Shape goes from (H, W) -> (H, W, 10)
        x_one_hot = F.one_hot(x_tensor, num_classes=10)
        
        # PyTorch Conv2d layers expect channels first: (Channels, H, W).
        # We permute the dimensions to shift the 10 channels to the front.
        # (Alternatively, you could use einops.rearrange(x_one_hot, 'h w c -> c h w') here).
        x_one_hot = x_one_hot.permute(2, 0, 1).to(torch.float32)
        
        # --- Y (Target) Transformation ---
        # Y is already binary (0 or 1), but the Binary Cross Entropy loss function 
        # expects a matching channel dimension.
        # Shape goes from (H, W) -> (1, H, W)
        y_tensor = torch.tensor(y_grid, dtype=torch.float32).unsqueeze(0)
        
        return x_one_hot, y_tensor
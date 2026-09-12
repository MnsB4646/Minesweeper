import torch
import torch.nn as nn

class MinesweeperCNN(nn.Module):
    """
    A Fully Convolutional Network (FCN) designed to predict mine probabilities.
    It uses 3x3 kernels with padding=1 to maintain the exact spatial dimensions
    of the board from input to output.
    """
    def __init__(self, in_channels=10, hidden_channels=64):
        super(MinesweeperCNN, self).__init__()
        
        # Stacking multiple convolutional layers increases the "receptive field".
        # 4 layers of 3x3 convolutions allow the network to "see" patterns 
        # up to 9 blocks away, which is enough to span a whole beginner board.
        self.network = nn.Sequential(
            # Layer 1: Extract basic features (edges, corners, number values)
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Layer 2: Combine features to find 1-2-1 or 1-2-2-1 logic patterns
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Layer 3: Deep spatial reasoning
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Layer 4: Deep spatial reasoning
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Output Layer: Collapse the 64 hidden channels down to 1 channel.
            # A 1x1 convolution acts as a final scoring mechanism for each specific cell.
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            
            # Sigmoid activation forces the raw output into a strict 0.0 to 1.0 probability
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (Batch, 10, Height, Width)
        Returns:
            Tensor: Probability heatmap of shape (Batch, 1, Height, Width)
        """
        return self.network(x)

# Quick mathematical test to verify tensor dimensions
if __name__ == "__main__":
    # Create a dummy batch of 4 beginner boards: (Batch=4, Channels=10, H=9, W=9)
    dummy_input = torch.zeros((4, 10, 9, 9))
    
    model = MinesweeperCNN()
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 1, 9, 9), "Output tensor dimensions are incorrect!"
    print("Model architecture compiled successfully.")
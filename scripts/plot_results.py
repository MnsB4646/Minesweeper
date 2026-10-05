import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path

# Add the project root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Create an 'assets' folder in the root directory to store the images
output_dir = Path(__file__).resolve().parent.parent / "assets"
output_dir.mkdir(exist_ok=True)

# Apply a clean, publication-ready style sheet
plt.style.use('ggplot')

def plot_depth_vs_winrate():
    """Bar chart showing the mathematical plateau of added layers."""
    layers = ['2 Layers', '3 Layers', '4 Layers', '5 Layers']
    win_rates = [34.20, 81.20, 86.20, 88.40]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(layers, win_rates, color='#4C72B0')
    
    plt.title('Ablation Study: Network Depth vs. Win Rate', fontsize=14, pad=15)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.ylim(0, 100)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.2f}%", ha='center', fontsize=11)
        
    plt.tight_layout()
    # Save the figure to the new assets directory
    plt.savefig(output_dir / 'depth_ablation.png', dpi=300, bbox_inches='tight')
    print(f"Saved depth_ablation.png to {output_dir.name}/")

def plot_width_overfitting():
    """Line graph demonstrating the overfitting threshold of network capacity."""
    channels = [8, 16, 64, 128]
    win_rates = [68.20, 87.60, 84.40, 85.00]
    
    plt.figure(figsize=(8, 5))
    plt.plot(channels, win_rates, marker='o', linewidth=2, markersize=8, color='#55A868')
    
    plt.title('Ablation Study: Network Width vs. Win Rate', fontsize=14, pad=15)
    plt.xlabel('Number of Channels', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.ylim(50, 100)
    plt.xticks(channels)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'width_ablation.png', dpi=300, bbox_inches='tight')
    print(f"Saved width_ablation.png to {output_dir.name}/")

def plot_data_volume_beginner():
    """Line graph showing how win rate scales with training data on Beginner."""
    # The x-axis strings allow the graph to space evenly despite the large numerical jumps
    games = ['1k', '5k', '10k', '50k']
    
    # TODO: Replace the placeholder values with your exact spreadsheet numbers 
    # (Leaving 94.0 for the 50k run as discussed)
    win_rates = [59.4, 85.8, 90.6, 94.0]
    
    plt.figure(figsize=(8, 5))
    plt.plot(games, win_rates, marker='s', linewidth=2, markersize=8, color='#C44E52')
    
    plt.title('Data Volume Scaling (Beginner Mode)', fontsize=14, pad=15)
    plt.xlabel('Training Games', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'data_volume_beginner.png', dpi=300, bbox_inches='tight')
    print(f"Saved data_volume_beginner.png to {output_dir.name}/")

def plot_data_volume_expert():
    """Line graph showing the difficulty of generalizing on Expert mode."""
    games = ['1k', '5k', '10k']
    
    win_rates = [1.2, 6.4, 12.0]
    
    plt.figure(figsize=(8, 5))
    plt.plot(games, win_rates, marker='^', linewidth=2, markersize=8, color='#8172B2')
    
    plt.title('Data Volume Scaling (Expert Mode)', fontsize=14, pad=15)
    plt.xlabel('Training Games', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    
    # Adjusted y-limit to better show the smaller values on Expert mode
    plt.ylim(0, 55) 
    
    plt.tight_layout()
    plt.savefig(output_dir / 'data_volume_expert.png', dpi=300, bbox_inches='tight')
    print(f"Saved data_volume_expert.png to {output_dir.name}/")

if __name__ == "__main__":
    plot_depth_vs_winrate()
    plot_width_overfitting()
    plot_data_volume_beginner()
    plot_data_volume_expert()
# Deep Learning Minesweeper AI

An advanced machine learning agent that plays Minesweeper by predicting mine probabilities using a Convolutional Neural Network (CNN).

Trained from scratch using self-play data, the model achieves a **94.0% win rate on Beginner boards**, operating within 2% of the absolute theoretical mathematical ceiling for perfect play (96.1%).

![Neural Agent Gameplay](assets/demo.gif)

## The Problem & Approach

Minesweeper is an NP-complete problem. While ~80% of a standard Beginner board can be solved using strict localized deduction rules, the remainder of the game requires statistical risk management and forced guessing.

Instead of hard-coding rule-based logic, this project utilizes a purely data-driven approach:

1. **Game State Representation:** The board is encoded into a $10 \times H \times W$ one-hot tensor (representing hidden, flagged, and numbers 0-8).
2. **Convolutional Processing:** A fully convolutional network processes the board, leveraging shared weights to learn translational invariance (a corner pattern is the same everywhere on the board).
3. **Probabilistic Output:** The network outputs a $1 \times H \times W$ heatmap of Sigmoid activations, representing the statistical probability of a mine at every cell. The agent simply clicks the cell with the lowest probability.

## Architecture & Ablation Studies

To determine the optimal architecture for the game, I conducted rigorous ablation studies focusing on network capacity, receptive fields, and data scaling.

### 1. The Receptive Field (Network Depth)

![Network Depth Ablation](assets/depth_ablation.png)

A standard 1-2-1 or 1-2-2-1 logic chain requires the AI to synthesize information across multiple tiles. I tested depth configurations from 2 to 5 layers.

* **Finding:** A 2-layer network ($5 \times 5$ receptive field) fails catastrophically (34% win rate) because it is physically blind to the edges of standard wall patterns. 4 layers ($9 \times 9$ receptive field) proved to be the optimal sweet spot (86.2%), allowing the agent to evaluate the entire width of a Beginner board from a center click.

### 2. Information Bottleneck vs. Overfitting (Network Width)

![Network Width Ablation](assets/width_ablation.png)

I tested the channel capacity (features learned per layer) to find the threshold of overfitting.

* **Finding:** 16 channels proved to be the most efficient and performant capacity (87.6%). Providing the network with 128 channels caused slight overfitting; the model memorized highly specific training board layouts, resulting in lower generalization and a slightly diminished win rate in live play, despite achieving lower validation loss.

### 3. Data Volume Scaling

![Beginner Data Volume](assets/data_volume_beginner.png)
![Expert Data Volume](assets/data_volume_expert.png)

To push the win rate past the "no-guess" limit of ~80%, the network required massive exposure to complex statistical variance.

* **Finding:** Scaling the self-play dataset to 50,000 games provided enough examples for the network to accurately minimize risk during ambiguous forced guesses, skyrocketing the win rate to 94.0%.
* **The Expert Mode Bottleneck:** When migrating to a $30 \times 16$ Expert board, performance drops significantly (6.4% at 5k games). This accurately highlights the limitation of a 4-layer CNN: Expert boards frequently require chording logic spanning 12+ tiles, which exists outside the $9 \times 9$ receptive field of the current architecture.

## Installation & Usage

### 1. Clone & Setup Environment

```bash
git clone https://github.com/yourusername/minesweeper-ai.git
cd minesweeper-ai
python -m venv .venv

# Windows activation
.venv\Scripts\activate
# Mac/Linux activation
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Play Against / Watch the AI

To launch an interactive game session where you can watch the pre-trained best model play:

```bash
python play.py
```

### 3. Train Your Own Model

Generate a new dataset and train the network from scratch. *Note: Ensure your `PYTHONPATH` allows module execution from the root directory as shown.*

```bash
# Generate 10,000 games of self-play data
python -m scripts.generate_data

# Train the network on the generated dataset
python -m scripts.train

# Benchmark the new checkpoint
python -m scripts.benchmark
```

## Repository Structure

```text
├── assets/                  # Generated graphs and demo GIFs
├── checkpoints/             # Contains best_model.pt (production weights)
├── scripts/                 # Execution pipelines (train, benchmark, data generation)
├── src/                     # Core logic
│   ├── agents/              # Neural, Random, and Rule-based agent classes
│   ├── engine/              # Game mechanics, board state, and enums
│   └── ml/                  # PyTorch model definitions and Dataset classes
└── tests/                   # Pytest suite for core engine mechanics
```

*Built as a research portfolio project demonstrating deep learning architecture design and ablation testing.*
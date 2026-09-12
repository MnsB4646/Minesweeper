import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.ml.dataset import MinesweeperDataset
from src.ml.model import MinesweeperCNN

# --- Hyperparameters & Config ---
DATASET_PATH = "minesweeper_dataset.npz"
CHECKPOINT_DIR = "checkpoints"
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
TRAIN_SPLIT = 0.8


def train():
    # 1. Device selection (Uses GPU acceleration if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Dataset and Train/Val split
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset '{DATASET_PATH}' not found. Run generate_data.py first!"
        )

    print("Loading dataset...")
    full_dataset = MinesweeperDataset(DATASET_PATH)
    total_samples = len(full_dataset)

    train_size = int(TRAIN_SPLIT * total_samples)
    val_size = total_samples - train_size

    # Deterministic generator seed ensures reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Total samples: {total_samples} (Train: {train_size}, Val: {val_size})")

    # 3. Initialize Model, Loss Function, and Optimizer
    model = MinesweeperCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")

    # 4. Training Loop
    print("\nStarting Training...\n" + "-" * 50)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_train_loss / train_size

        # 5. Validation Evaluation
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                running_val_loss += loss.item() * batch_x.size(0)

        epoch_val_loss = running_val_loss / val_size

        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}",
            end="",
        )

        # Checkpoint: Save model if validation loss improved
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(" -> [Saved New Best Model]")
        else:
            print()

    print("-" * 50)
    print(f"Training Complete! Best model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
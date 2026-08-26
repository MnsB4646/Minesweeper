from enum import Enum

class GameState(Enum):
    """Represents the current high-level state of the game."""
    ONGOING = 1
    WON = 2
    LOST = 3
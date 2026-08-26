import random
from .base import BaseAgent

class RandomAgent(BaseAgent):
    """A naive bot that clicks unrevealed cells completely at random."""

    def select_action(self, board, revealed_set, flags_set):
        height = len(board)
        width = len(board[0])
        
        valid_moves = []
        for y in range(height):
            for x in range(width):
                if (x, y) not in revealed_set and (x, y) not in flags_set:
                    valid_moves.append((x, y))
                    
        return random.choice(valid_moves) if valid_moves else (0, 0)
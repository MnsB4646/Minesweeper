from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Blueprint that all Minesweeper bots must follow."""
    
    @abstractmethod
    def select_action(self, board, revealed_set, flags_set):
        """
        Evaluates the board and returns an (x, y) coordinate to click.
        """
        pass
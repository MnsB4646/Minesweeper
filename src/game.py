from .enums import GameState
from .board import Board

class Game:
    """
    Manages the high-level game loop, win/loss conditions, and acts as the 
    main API.
    """

    def __init__(self, width, height, num_mines):
        """
        Initializes a new game session.

        Args:
            width (int): Columns in the grid.
            height (int): Rows in the grid.
            num_mines (int): Total hidden mines.
        """
        self.board = Board(width, height, num_mines)
        self.state = GameState.ONGOING
        self.first_click = True

    def click(self, x, y):
        """
        Executes a click at the specified coordinates and updates the game state.

        On the very first click, it populates the board to ensure the player 
        does not hit a mine immediately.

        Args:
            x (int): The x-coordinate (column).
            y (int): The y-coordinate (row).

        Returns:
            GameState: The current state of the game (ONGOING, WON, or LOST).
        """
        if self.state != GameState.ONGOING:
            return self.state

        # Defer mine placement until the first click to guarantee safety
        if self.first_click:
            self.board.place_mines(x, y)
            self.board.calculate_adjacent_counts()
            self.first_click = False

        try:
            self.board.reveal(x, y)
        except Exception:
            self.state = GameState.LOST
            return self.state

        self._check_win()
        return self.state
    
    def flag(self, x, y):
        """
        Toggles a flag on the specified coordinate if the game is ongoing.
        """
        if self.state == GameState.ONGOING:
            self.board.flag(x, y)

    def _check_win(self):
        """
        Evaluates if the victory condition has been met.
        A win occurs when the number of revealed cells equals the total 
        number of non-mine cells on the board.
        """
        total_cells = self.board.width * self.board.height
        safe_cells = total_cells - self.board.num_mines
        
        if len(self.board.revealed) == safe_cells:
            self.state = GameState.WON
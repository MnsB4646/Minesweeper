class Board:
    """
    Represents the core game state and logic for a Minesweeper environment.
    """

    def __init__(self, width, height, num_mines):
        """
        Initializes the Minesweeper board memory and state trackers.

        Args:
            width (int): The number of columns in the grid.
            height (int): The number of rows in the grid.
            num_mines (int): The total number of mines to place on the board.
        """
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.board = [[0 for _ in range(width)] for _ in range(height)]
        self.mines = set()
        self.revealed = set()
        self.flags = set()

    def reveal(self, x, y):
        """
        Reveals a cell at the given coordinates.

        If the cell has no adjacent mines (value is 0), it recursively reveals
        all adjacent cells using a flood-fill algorithm.

        Args:
            x (int): The x-coordinate (column) of the cell to reveal.
            y (int): The y-coordinate (row) of the cell to reveal.

        Raises:
            Exception: If the revealed cell contains a mine (value of -1).
        """
        if (x, y) in self.revealed or (x, y) in self.flags:
            return
        
        self.revealed.add((x, y))
        
        if self.board[y][x] == 0:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        self.reveal(nx, ny)
                        
        if self.board[y][x] == -1:
            raise Exception("Game Over! You hit a mine.")

    def flag(self, x, y):
        """
        Toggles a flag on or off for a specific cell.

        Flags mark suspected mines and protect cells from being accidentally 
        revealed. Revealed cells cannot be flagged.

        Args:
            x (int): The x-coordinate (column) of the cell to flag or unflag.
            y (int): The y-coordinate (row) of the cell to flag or unflag.
        """
        if (x, y) in self.revealed:
            return
        if (x, y) in self.flags:
            self.flags.remove((x, y))
        else:
            self.flags.add((x, y))

    def place_mines(self, x1, y1):
        """
        Randomly populates the board with mines.

        Ensures that the player's first clicked coordinate (x1, y1) is never 
        a mine to guarantee a safe first move. Updates the underlying board 
        array with -1 for mine locations.

        Args:
            x1 (int): The x-coordinate of the first clicked cell.
            y1 (int): The y-coordinate of the first clicked cell.
        """
        import random
        while len(self.mines) < self.num_mines:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.mines and (x, y) not in self.revealed and (x, y) != (x1, y1):
                self.mines.add((x, y))
                self.board[y][x] = -1  # -1 represents a mine

    def calculate_adjacent_counts(self):
        """
        Calculates and updates the adjacent mine counts for all non-mine cells.

        Iterates through the board and sets the integer value of each safe cell 
        to the total number of mines (0-8) present in its immediate 3x3 neighborhood.
        """
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == -1:
                    continue
                count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.height and 0 <= nx < self.width:
                            if self.board[ny][nx] == -1:
                                count += 1
                self.board[y][x] = count
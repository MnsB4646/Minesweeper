import random
from .base import BaseAgent

class GoodAgent(BaseAgent):
    """
    A heuristic bot that uses a two-pass constraint satisfaction algorithm 
    to deduce safe moves, falling back to random guessing only when necessary.
    """

    def select_action(self, board, revealed_set, flags_set):
        height = len(board)
        width = len(board[0])
        
        def get_neighbors(cx, cy):
            """Helper to grab all valid adjacent coordinates."""
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbors.append((nx, ny))
            return neighbors

        # The agent's internal memory of where mines must be
        known_mines = set(flags_set)
        safe_moves = set()

        # PASS 1: Deduce guaranteed mines
        for y in range(height):
            for x in range(width):
                if (x, y) in revealed_set and board[y][x] > 0:
                    neighbors = get_neighbors(x, y)
                    unrevealed = [n for n in neighbors if n not in revealed_set]
                    
                    # If unrevealed neighbors exactly match the cell's number, all are mines
                    if len(unrevealed) == board[y][x]:
                        for n in unrevealed:
                            known_mines.add(n)

        # PASS 2: Deduce guaranteed safe cells
        for y in range(height):
            for x in range(width):
                if (x, y) in revealed_set and board[y][x] > 0:
                    neighbors = get_neighbors(x, y)
                    unrevealed = [n for n in neighbors if n not in revealed_set]
                    
                    # Count how many of these unrevealed neighbors we just identified as mines
                    adjacent_mines = sum(1 for n in unrevealed if n in known_mines)
                    
                    # If we account for all mines around this number, the remaining cells are safe
                    if adjacent_mines == board[y][x]:
                        for n in unrevealed:
                            if n not in known_mines:
                                safe_moves.add(n)

        # Execute a known safe move if one exists
        if safe_moves:
            return random.choice(list(safe_moves))

        # PASS 3: If forced to guess, pick a cell we don't already know is a mine
        guessing_moves = []
        for y in range(height):
            for x in range(width):
                if (x, y) not in revealed_set and (x, y) not in known_mines:
                    guessing_moves.append((x, y))

        if guessing_moves:
            return random.choice(guessing_moves)
            
        return (0, 0)
import sys
import pygame
from src.game import Game
from src.enums import GameState
from src.agents.random_agent import RandomAgent
from src.agents.good_agent import GoodAgent

# --- Configuration & Colors ---
CELL_SIZE = 40
FPS = 60
MOVE_DELAY_MS = 500  # How long to wait between bot moves

COLOR_HIDDEN = (180, 180, 180)
COLOR_REVEALED = (230, 230, 230)
COLOR_MINE = (255, 100, 100)
COLOR_LINE = (100, 100, 100)
COLOR_TEXT = (30, 30, 30)

def draw_board(screen, font, game):
    """Renders the current game state to the Pygame screen."""
    board = game.board
    screen.fill(COLOR_LINE)

    for y in range(board.height):
        for x in range(board.width):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            
            if (x, y) in board.revealed:
                val = board.board[y][x]
                if val == -1:
                    pygame.draw.rect(screen, COLOR_MINE, rect)
                    text = font.render("*", True, COLOR_TEXT)
                else:
                    pygame.draw.rect(screen, COLOR_REVEALED, rect)
                    text = font.render(str(val) if val > 0 else "", True, COLOR_TEXT)
            else:
                pygame.draw.rect(screen, COLOR_HIDDEN, rect)
                text = font.render("F" if (x, y) in board.flags else "", True, COLOR_TEXT)

            # Draw cell borders
            pygame.draw.rect(screen, COLOR_LINE, rect, 1)
            
            # Center text in the cell
            if text:
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

def play_and_watch(agent, width=9, height=9, mines=20):
    """Main graphical loop."""
    pygame.init()
    
    # Setup window and typography
    screen = pygame.display.set_mode((width * CELL_SIZE, height * CELL_SIZE))
    pygame.display.set_caption("Minesweeper AI Visualizer")
    font = pygame.font.SysFont("arial", int(CELL_SIZE * 0.6), bold=True)
    clock = pygame.time.Clock()
    
    game = Game(width, height, mines)
    
    running = True
    last_move_time = pygame.time.get_ticks()

    while running:
        # 1. Handle window events (like clicking the X to close)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 2. Agent Logic (Non-blocking timer)
        current_time = pygame.time.get_ticks()
        if game.state == GameState.ONGOING and (current_time - last_move_time > MOVE_DELAY_MS):
            x, y = agent.select_action(game.board.board, game.board.revealed, game.board.flags)
            game.click(x, y)
            last_move_time = current_time

            # Print game over status to the terminal
            if game.state != GameState.ONGOING:
                print(f"Game Over! Result: {game.state.name}")

        # 3. Render Graphics
        draw_board(screen, font, game)
        pygame.display.flip()
        
        # 4. Cap the framerate to save CPU
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    bot = GoodAgent()
    play_and_watch(bot)
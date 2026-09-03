import time
from src.game import Game
from src.enums import GameState
from src.agents.random_agent import RandomAgent
from src.agents.good_agent import GoodAgent # Assuming you saved it here

def run_benchmark(agent, num_games=100, width=20, height=20, num_mines=50):
    """
    Runs a headless simulation of the game and calculates performance statistics.
    """
    print(f"Starting benchmark for {agent.__class__.__name__} ({num_games} games)...")
    
    wins = 0
    total_moves = 0
    total_completion_percentage = 0.0
    
    start_time = time.time()

    for _ in range(num_games):
        game = Game(width, height, num_mines)
        moves = 0
        
        while game.state == GameState.ONGOING:
            # The agent decides
            x, y = agent.select_action(game.board.board, game.board.revealed, game.board.flags)
            
            # The game advances
            game.click(x, y)
            moves += 1

        # Record statistics for this specific game
        total_moves += moves
        if game.state == GameState.WON:
            wins += 1
            total_completion_percentage += 100.0
        else:
            # Calculate how much of the board was successfully solved before dying
            total_cells = width * height
            safe_cells = total_cells - num_mines
            revealed_count = len(game.board.revealed)
            completion = (revealed_count / safe_cells) * 100
            total_completion_percentage += completion

    # Calculate final aggregate metrics
    end_time = time.time()
    time_taken = end_time - start_time
    
    win_rate = (wins / num_games) * 100
    avg_moves = total_moves / num_games
    avg_completion = total_completion_percentage / num_games
    games_per_second = num_games / time_taken if time_taken > 0 else 0

    # Print the scorecard
    print("-" * 40)
    print(f"Results for {agent.__class__.__name__}:")
    print(f"Games Played:    {num_games}")
    print(f"Win Rate:        {win_rate:.2f}%")
    print(f"Avg Completion:  {avg_completion:.2f}% of safe cells")
    print(f"Avg Moves/Game:  {avg_moves:.1f}")
    print(f"Time Taken:      {time_taken:.2f} seconds ({games_per_second:.0f} games/sec)")
    print("-" * 40)
    print()

if __name__ == "__main__":
    # Benchmark the dummy bot
    random_bot = RandomAgent()
    run_benchmark(random_bot, num_games=1000)
    
    # Benchmark the smart bot
    good_bot = GoodAgent()
    run_benchmark(good_bot, num_games=1000)
from unittest.mock import Mock, patch

import pytest

from src.board import Board
from src.enums import GameState
from src.game import Game


def test_init_creates_board_and_starts_ongoing():
    game = Game(4, 3, 2)

    assert isinstance(game.board, Board)
    assert (game.board.width, game.board.height, game.board.num_mines) == (4, 3, 2)
    assert game.state == GameState.ONGOING
    assert game.first_click is True


def test_first_click_places_mines_calculates_counts_and_reveals():
    game = Game(2, 2, 1)
    game.board.place_mines = Mock()
    game.board.calculate_adjacent_counts = Mock()
    game.board.reveal = Mock()
    game._check_win = Mock()

    result = game.click(1, 0)

    assert result == GameState.ONGOING
    assert game.first_click is False
    game.board.place_mines.assert_called_once_with(1, 0)
    game.board.calculate_adjacent_counts.assert_called_once_with()
    game.board.reveal.assert_called_once_with(1, 0)
    game._check_win.assert_called_once_with()


def test_later_click_skips_initialization_and_checks_win():
    game = Game(2, 1, 1)
    game.first_click = False
    game.board.place_mines = Mock()
    game.board.calculate_adjacent_counts = Mock()
    game.board.reveal = Mock()
    game._check_win = Mock()

    result = game.click(0, 0)

    assert result == GameState.ONGOING
    game.board.place_mines.assert_not_called()
    game.board.calculate_adjacent_counts.assert_not_called()
    game.board.reveal.assert_called_once_with(0, 0)
    game._check_win.assert_called_once_with()


def test_click_returns_lost_when_reveal_hits_a_mine():
    game = Game(1, 1, 1)
    game.board.place_mines = Mock()
    game.board.calculate_adjacent_counts = Mock()
    game.board.reveal = Mock(side_effect=Exception("mine"))

    result = game.click(0, 0)

    assert result == GameState.LOST
    assert game.state == GameState.LOST
    assert game.first_click is False


def test_click_after_game_over_does_nothing():
    game = Game(1, 1, 0)
    game.state = GameState.WON
    game.board.reveal = Mock()

    result = game.click(0, 0)

    assert result == GameState.WON
    game.board.reveal.assert_not_called()
    assert game.first_click is True


def test_flag_delegates_while_game_is_ongoing():
    game = Game(2, 1, 0)
    game.board.flag = Mock()

    result = game.flag(1, 0)

    assert result is None
    game.board.flag.assert_called_once_with(1, 0)


@pytest.mark.parametrize("state", [GameState.WON, GameState.LOST])
def test_flag_does_nothing_after_game_is_over(state):
    game = Game(2, 1, 0)
    game.state = state
    game.board.flag = Mock()

    result = game.flag(1, 0)

    assert result is None
    game.board.flag.assert_not_called()


def test_check_win_sets_won_when_all_safe_cells_are_revealed():
    game = Game(3, 2, 2)
    game.board.revealed = {(0, 0), (1, 0), (2, 0), (0, 1)}

    game._check_win()

    assert game.state == GameState.WON


def test_check_win_keeps_game_ongoing_until_all_safe_cells_are_revealed():
    game = Game(3, 2, 2)
    game.board.revealed = {(0, 0), (1, 0), (2, 0)}

    game._check_win()

    assert game.state == GameState.ONGOING


def test_click_can_transition_to_won_on_a_board_without_mines():
    game = Game(2, 1, 0)

    with patch.object(game.board, "place_mines"), patch.object(
        game.board, "calculate_adjacent_counts"
    ):
        result = game.click(0, 0)

    assert result == GameState.WON
    assert game.state == GameState.WON

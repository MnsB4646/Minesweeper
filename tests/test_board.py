import sys
from pathlib import Path
from unittest.mock import patch

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from board import Board


def test_init_creates_empty_board_and_state():
    board = Board(3, 2, 1)

    assert board.width == 3
    assert board.height == 2
    assert board.num_mines == 1
    assert board.board == [[0, 0, 0], [0, 0, 0]]
    assert board.mines == set()
    assert board.revealed == set()
    assert board.flags == set()


def test_reveal_reveals_cell_and_does_not_reveal_adjacent_nonzero_cells():
    board = Board(3, 1, 0)
    board.board[0] = [1, 2, 1]

    board.reveal(1, 0)

    assert board.revealed == {(1, 0)}


def test_reveal_flood_fills_zero_cells():
    board = Board(3, 2, 0)

    board.reveal(0, 0)

    assert board.revealed == {
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
    }


def test_reveal_ignores_revealed_and_flagged_cells():
    board = Board(2, 1, 0)
    board.board[0] = [1, 1]
    board.revealed.add((0, 0))
    board.flags.add((1, 0))

    board.reveal(0, 0)
    board.reveal(1, 0)

    assert board.revealed == {(0, 0)}
    assert board.flags == {(1, 0)}


def test_reveal_raises_when_cell_contains_a_mine():
    board = Board(1, 1, 1)
    board.board[0][0] = -1

    with pytest.raises(Exception, match="Game Over! You hit a mine."):
        board.reveal(0, 0)

    assert board.revealed == {(0, 0)}


def test_flag_toggles_flag_for_unrevealed_cell():
    board = Board(2, 1, 0)

    board.flag(1, 0)
    assert board.flags == {(1, 0)}

    board.flag(1, 0)
    assert board.flags == set()


def test_flag_does_not_change_revealed_cells():
    board = Board(1, 1, 0)
    board.revealed.add((0, 0))

    board.flag(0, 0)

    assert board.flags == set()


def test_place_mines_places_requested_count_and_keeps_first_cell_safe():
    board = Board(3, 2, 2)

    with patch("random.randint", side_effect=[1, 1, 0, 0, 0, 0, 2, 1]):
        board.place_mines(1, 1)

    assert board.mines == {(0, 0), (2, 1)}
    assert board.board == [[-1, 0, 0], [0, 0, -1]]
    assert (1, 1) not in board.mines


def test_place_mines_does_not_place_mines_on_revealed_cells():
    board = Board(3, 1, 1)
    board.revealed.add((0, 0))

    with patch("random.randint", side_effect=[0, 0, 2, 0, 1, 0]):
        board.place_mines(2, 0)

    assert board.mines == {(1, 0)}
    assert board.board == [[0, -1, 0]]


def test_calculate_adjacent_counts_updates_safe_cells_and_preserves_mines():
    board = Board(3, 3, 2)
    board.board = [
        [-1, 0, 0],
        [0, 0, 0],
        [0, 0, -1],
    ]

    board.calculate_adjacent_counts()

    assert board.board == [
        [-1, 1, 0],
        [1, 2, 1],
        [0, 1, -1],
    ]

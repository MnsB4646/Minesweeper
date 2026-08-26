from unittest.mock import patch

import pytest



from src.board import Board


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
    board = Board(5, 5, 2)

    with patch("random.randint", side_effect=[2, 2, 1, 1, 4, 4, 0, 0]):
        board.place_mines(2, 2)

    assert board.mines == {(4, 4), (0, 0)}
    assert board.board[4][4] == -1
    assert board.board[0][0] == -1
    assert (2, 2) not in board.mines


def test_place_mines_keeps_all_neighbors_of_first_click_safe():
    board = Board(5, 5, 1)
    invalid_coordinates = [
        (x, y)
        for y in range(1, 4)
        for x in range(1, 4)
        if (x, y) != (2, 2)
    ]
    random_values = [coordinate for point in invalid_coordinates for coordinate in point]
    random_values.extend([0, 0])

    with patch("random.randint", side_effect=random_values):
        board.place_mines(2, 2)

    neighbors = {
        (x, y)
        for x in range(1, 4)
        for y in range(1, 4)
        if (x, y) != (2, 2)
    }
    assert board.mines == {(0, 0)}
    assert board.mines.isdisjoint(neighbors | {(2, 2)})


def test_place_mines_does_not_place_mines_on_revealed_cells():
    board = Board(5, 3, 1)
    board.revealed.add((0, 0))

    with patch("random.randint", side_effect=[0, 0, 1, 1, 4, 2]):
        board.place_mines(2, 0)

    assert board.mines == {(4, 2)}
    assert board.board[2][4] == -1


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

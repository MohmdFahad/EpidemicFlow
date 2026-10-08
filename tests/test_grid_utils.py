import numpy as np
import pytest

from epidemicflow.grid_utils import INFECTED, get_neighbours, get_pos, get_score, make_grid


@pytest.mark.parametrize("pos, expected_neighbours", [((2, 2), 8), ((0, 2), 5), ((0, 0), 3)])
def test_neighbour_counts(pos, expected_neighbours):
    assert len(get_neighbours(np.array(pos), 5)) == expected_neighbours


@pytest.mark.parametrize("pos", [(2, 2), (0, 2), (0, 0)])  # interior, edge, corner
def test_score_is_one_when_all_neighbours_match(pos):
    grid = np.ones((5, 5), dtype=np.int8)
    assert get_score(grid, get_neighbours(np.array(pos), 5), 1) == pytest.approx(1.0)


def test_score_counts_multiple_labels():
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[0, 0], grid[0, 1] = 4, 5
    assert get_score(grid, get_neighbours(np.array((1, 1)), 3), 4, 5) == pytest.approx(2 / 8)


def test_get_pos_finds_all_positions():
    grid = np.zeros((4, 4), dtype=np.int8)
    grid[1, 2] = grid[3, 0] = 5
    assert sorted(map(tuple, get_pos(grid, 5))) == [(1, 2), (3, 0)]
    assert get_pos(grid, 6).shape == (0, 2)


def test_make_grid_places_exact_number_of_infections():
    pop = make_grid(10, 7, np.random.default_rng(0))
    assert (pop.grid == INFECTED).sum() == 7
    assert (pop.infection_day_grid == 0).sum() == 7


def test_make_grid_rejects_too_many_infections():
    with pytest.raises(ValueError):
        make_grid(3, 10, np.random.default_rng(0))

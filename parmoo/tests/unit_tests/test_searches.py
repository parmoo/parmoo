""" Unit tests for the parmoo.searches plugins. """

import numpy as np
import pytest

from parmoo.searches import LatinHypercube

LB = -1.0 * np.ones(5)
UB = np.zeros(5)


def test_LatinHypercube_bad_search_budget():
    """ Check the 'search_budget' hyperparameter contract. """

    with pytest.raises(TypeError):
        LatinHypercube(2, LB, UB, {'search_budget': 2.0})
    with pytest.raises(ValueError):
        LatinHypercube(2, LB, UB, {'search_budget': -1})


def test_LatinHypercube_empty_budget():
    """ Check that a search budget of zero produces an empty design.

    Both the initial search and a resumed search must return nothing.

    """

    search = LatinHypercube(2, LB, UB, {'search_budget': 0})
    assert (np.size(search.startSearch(LB, UB)) == 0)
    assert (np.size(search.resumeSearch()) == 0)


def test_LatinHypercube_design_shape_and_bounds():
    """ Check that the design has the requested shape and respects the bounds.

    A resumed search must produce a second design of the same shape, also
    within the bounds.

    """

    search = LatinHypercube(2, LB, UB, {})
    for design in [search.startSearch(LB, UB), search.resumeSearch()]:
        assert (np.shape(design) == (100, 5))
        assert (all([all(xi <= UB) and all(xi >= LB) for xi in design]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

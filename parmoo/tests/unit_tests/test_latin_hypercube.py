
def test_LatinHypercube():
    """ Test the LatinHypercube class in searches.py.

    Use the LatinHypercube class to generate a LatinHypercube in 5 dimensions,
    check that the dimensions of the design agree with the input, then
    generate and check another design.

    """

    from parmoo.searches import LatinHypercube
    import numpy as np
    import pytest

    # Set the dimensions for the design
    lb = -1.0 * np.ones(5)
    ub = np.zeros(5)
    # Try to initialize a bad search to test error handling
    with pytest.raises(ValueError):
        LatinHypercube(2, lb, ub, {'search_budget': 2.0})
    with pytest.raises(ValueError):
        LatinHypercube(2, lb, ub, {'search_budget': -1})
    # Create a good LHS object
    latin_search = LatinHypercube(2, lb, ub, {'search_budget': 0})
    # Generate a design of size 0 and confirm that it is an empty array
    des1 = latin_search.startSearch(lb, ub)
    assert (np.size(des1) == 0)
    # Resume the search with size 0 and check that it is an empty array
    des2 = latin_search.resumeSearch()
    assert (np.size(des2) == 0)
    # Generate a new design and check that it conforms to the dimensions
    latin_search = LatinHypercube(2, lb, ub, {})
    des3 = latin_search.startSearch(lb, ub)
    assert (np.shape(des3) == (100, 5))
    assert (all([all(xi <= ub) and all(xi >= lb) for xi in des3]))
    # Resume the search and check that it conforms to the dimensions
    des4 = latin_search.resumeSearch()
    assert (np.shape(des4) == (100, 5))
    assert (all([all(xi <= ub) and all(xi >= lb) for xi in des4]))
    return


if __name__ == "__main__":
    test_LatinHypercube()


def test_AcquisitionFunction():
    """ Test that the AcquisitionFunction ABC raises a TypeError"""

    from parmoo.acquisitions.acquisition_function import AcquisitionFunction
    import pytest

    with pytest.raises(TypeError):
        AcquisitionFunction(0, 0, 0, 0)


def test_GlobalSearch():
    """ Test that the GlobalSearch ABC raises a TypeError"""

    from parmoo.searches.global_search import GlobalSearch
    import pytest

    with pytest.raises(TypeError):
        GlobalSearch(0, 0, 0, 0)


def test_SurrogateFunction():
    """ Test that the SurrogateFunction ABC raises a TypeError"""

    from parmoo.surrogates.surrogate_function import SurrogateFunction
    import pytest

    with pytest.raises(TypeError):
        SurrogateFunction(0, 0, 0, 0)


def test_SurrogateOptimizer():
    """ Test that the SurrogateFunction ABC raises a TypeError"""

    from parmoo.optimizers.surrogate_optimizer import SurrogateOptimizer
    import pytest

    with pytest.raises(TypeError):
        SurrogateOptimizer(0, 0, 0, 0)


if __name__ == "__main__":
    test_AcquisitionFunction()
    test_GlobalSearch()
    test_SurrogateFunction()
    test_SurrogateOptimizer()

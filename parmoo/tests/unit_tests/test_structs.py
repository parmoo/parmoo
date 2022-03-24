
def test_AcquisitionFunction():
    """ Test that the AcquisitionFunction ABC raises NotImplementedError()"""

    from parmoo.structs import AcquisitionFunction
    import pytest

    with pytest.raises(NotImplementedError):
        AcquisitionFunction(0, 0, 0, 0)
    with pytest.raises(NotImplementedError):
        AcquisitionFunction.setTarget(0, 0, 0, 0)
    with pytest.raises(NotImplementedError):
        AcquisitionFunction.scalarize(0, 0)
    with pytest.raises(NotImplementedError):
        AcquisitionFunction.scalarizeGrad(0, 0, 0)
    with pytest.raises(NotImplementedError):
        AcquisitionFunction.save(0, 0)
    with pytest.raises(NotImplementedError):
        AcquisitionFunction.load(0, 0)


def test_GlobalSearch():
    """ Test that the GlobalSearch ABC raises NotImplementedError()"""

    from parmoo.structs import GlobalSearch
    import pytest

    with pytest.raises(NotImplementedError):
        GlobalSearch(0, 0, 0, 0)
    with pytest.raises(NotImplementedError):
        GlobalSearch.startSearch(0, 0, 0)
    with pytest.raises(NotImplementedError):
        GlobalSearch.resumeSearch(0)
    with pytest.raises(NotImplementedError):
        GlobalSearch.save(0, 0)
    with pytest.raises(NotImplementedError):
        GlobalSearch.load(0, 0)


def test_SurrogateFunction():
    """ Test that the SurrogateFunction ABC raises NotImplementedError()"""

    from parmoo.structs import SurrogateFunction
    import pytest

    with pytest.raises(NotImplementedError):
        SurrogateFunction(0, 0, 0, 0)
    with pytest.raises(NotImplementedError):
        SurrogateFunction.fit(0, 0, 0)
    with pytest.raises(NotImplementedError):
        SurrogateFunction.update(0, 0, 0)
    with pytest.raises(NotImplementedError):
        SurrogateFunction.setCenter(0, 0)
    with pytest.raises(NotImplementedError):
        SurrogateFunction.evaluate(0, 0)
    with pytest.raises(NotImplementedError):
        SurrogateFunction.gradient(0, 0)
    with pytest.raises(NotImplementedError):
        SurrogateFunction.improve(0, 0, 0)
    with pytest.raises(NotImplementedError):
        SurrogateFunction.save(0, 0)
    with pytest.raises(NotImplementedError):
        SurrogateFunction.load(0, 0)


def test_SurrogateOptimizer():
    """ Test that the SurrogateFunction ABC raises NotImplementedError()"""

    from parmoo.structs import SurrogateOptimizer
    import pytest

    with pytest.raises(NotImplementedError):
        SurrogateOptimizer(0, 0, 0, 0)
    with pytest.raises(NotImplementedError):
        SurrogateOptimizer.setObjective(0, 0)
    with pytest.raises(NotImplementedError):
        SurrogateOptimizer.setGradient(0, 0)
    with pytest.raises(NotImplementedError):
        SurrogateOptimizer.setConstraints(0, 0)
    with pytest.raises(NotImplementedError):
        SurrogateOptimizer.addAcquisition(0, 0)
    with pytest.raises(NotImplementedError):
        SurrogateOptimizer.setReset(0, 0)
    with pytest.raises(NotImplementedError):
        SurrogateOptimizer.solve(0, 0)
    with pytest.raises(NotImplementedError):
        SurrogateOptimizer.save(0, 0)
    with pytest.raises(NotImplementedError):
        SurrogateOptimizer.load(0, 0)


if __name__ == "__main__":
    test_AcquisitionFunction()
    test_GlobalSearch()
    test_SurrogateFunction()
    test_SurrogateOptimizer()

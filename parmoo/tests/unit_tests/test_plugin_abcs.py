""" Unit tests for ParMOO's plugin abstract base classes.

Each extension point is an ABC with abstract methods, so none of them may be
instantiated directly.  All four share the (dim, lb, ub, hyperparams)
constructor signature that ParMOO's checkers probe plugin classes with.

"""

import pytest

from parmoo.acquisitions.acquisition_function import AcquisitionFunction
from parmoo.optimizers.surrogate_optimizer import SurrogateOptimizer
from parmoo.searches.global_search import GlobalSearch
from parmoo.surrogates.surrogate_function import SurrogateFunction

PLUGIN_ABCS = [
    AcquisitionFunction,
    GlobalSearch,
    SurrogateFunction,
    SurrogateOptimizer,
]


@pytest.mark.parametrize("cls", PLUGIN_ABCS)
def test_plugin_abc_is_not_instantiable(cls):
    """ Check that each plugin ABC refuses to be instantiated. """

    with pytest.raises(TypeError):
        cls(0, 0, 0, 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

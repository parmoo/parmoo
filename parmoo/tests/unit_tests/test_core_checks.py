""" Unit tests for parmoo.core.moop_checks.

check_sims() validates the whole argument dict passed to MOOP.addSimulation(),
so this is the single place that the simulation dict contract is tested.

"""

import numpy as np
import pytest

from parmoo.core.moop_checks import check_sims
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF


def test_check_sims():
    """ Check that the check_sims() utility handles bad input correctly.

    Provide several bad simulation dictionaries to check_sims() and
    confirm that it raises the appropriate ValueErrors.

    """

    # Try providing invalid/incompatible simulation dictionaries
    with pytest.raises(TypeError):
        check_sims(3, 5.0)
    simdict = {}
    with pytest.raises(KeyError):
        check_sims(3, simdict)
    simdict['m'] = 1.0
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['m'] = -1
    with pytest.raises(ValueError):
        check_sims(3, simdict)
    simdict['m'] = 1
    with pytest.raises(KeyError):
        check_sims(3, simdict)
    simdict['search'] = LatinHypercube(1, np.zeros(3), np.ones(3), {})
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['search'] = GaussRBF
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['search'] = LatinHypercube
    with pytest.raises(KeyError):
        check_sims(3, simdict)
    simdict['surrogate'] = GaussRBF(1, np.zeros(1), np.ones(1), {})
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['surrogate'] = LatinHypercube
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['surrogate'] = GaussRBF
    with pytest.raises(KeyError):
        check_sims(3, simdict)
    simdict['sim_func'] = {}
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['sim_func'] = lambda x, y, z: [np.linalg.norm(x - y - z)]
    with pytest.raises(ValueError):
        check_sims(3, simdict)
    simdict['sim_func'] = lambda x: [np.linalg.norm(x)]
    simdict['hyperparams'] = []
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['hyperparams'] = {}
    simdict['sim_db'] = 5
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['sim_db'] = {'x_vals': []}
    with pytest.raises(KeyError):
        check_sims(3, simdict)
    simdict['sim_db'] = {'x_vals': "hel", 's_vals': "lo"}
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['sim_db'] = {'x_vals': np.asarray([0.0]), 's_vals': []}
    with pytest.raises(ValueError):
        check_sims(3, simdict)
    simdict['sim_db'] = {'x_vals': np.asarray([[0.0, 0.0]]), 's_vals': [[0.0]]}
    with pytest.raises(ValueError):
        check_sims(3, simdict)
    simdict['sim_db'] = {'x_vals': np.asarray([[0.0, 0.0, 0.0]]),
                         's_vals': [[0.0, 0.0]]}
    with pytest.raises(ValueError):
        check_sims(3, simdict)
    simdict['sim_db'] = {'x_vals': np.asarray([[0.0, 0.0, 0.0]]),
                         's_vals': [[0.0], [1.0]]}
    with pytest.raises(ValueError):
        check_sims(3, simdict)
    simdict['sim_db'] = {'x_vals': [], 's_vals': []}
    simdict['des_tol'] = 1
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['des_tol'] = 0.0
    with pytest.raises(ValueError):
        check_sims(3, simdict)
    simdict['des_tol'] = 0.00000001
    simdict['name'] = 5
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    # Do one good check, and make sure nothing was raised
    simdict['name'] = "sim1"
    check_sims(3, simdict)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

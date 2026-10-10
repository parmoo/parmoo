""" Unit tests for the DTLZ simulation and objective libraries.

The DTLZ problems share a common structure: a kernel function g that measures
the distance to the Pareto front, and a set of objectives that spread points
along it.  Every test here is a single evaluation against a hand-computed
value, so they are expressed as parametrized tables rather than one function
per problem.

"""

import numpy as np
import pytest

from parmoo.objectives.dtlz import (
    dtlz1_obj,
    dtlz2_obj,
    dtlz3_obj,
    dtlz4_obj,
)
from parmoo.simulations.dtlz import (
    dtlz1_sim,
    dtlz2_sim,
    dtlz3_sim,
    dtlz4_sim,
    dtlz5_sim,
    dtlz6_sim,
    dtlz7_sim,
    dtlz8_sim,
    dtlz9_sim,
    g1_sim,
    g2_sim,
    g3_sim,
    g4_sim,
)

# The 5-variable design space used for the simulations, evaluated at the point
# where every variable equals the offset, so each kernel is at its minimum.
SIM_DTYPE = [(f"x{i + 1}", "f8") for i in range(5)]
OFFSET = 0.6
AT_OFFSET = {f"x{i + 1}": OFFSET for i in range(5)}

# The 4-variable design space used for the objectives, evaluated at the origin
OBJ_XTYPE = [(f"x{i + 1}", "f8") for i in range(4)]
OBJ_STYPE = [("sim1", "f8")]
NUM_OBJ = 3
TOL = 1.0e-4


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel, expected", [
    # g1, g2, and g3 all vanish when every variable sits at the offset
    (g1_sim, 0.0),
    (g2_sim, 0.0),
    (g3_sim, 0.0),
    # g4 is normalized differently and reaches one there instead
    (g4_sim, 1.0),
])
def test_dtlz_kernel_at_offset(kernel, expected):
    """ Check each DTLZ kernel function at its minimizer. """

    g = kernel(SIM_DTYPE, num_obj=NUM_OBJ, offset=OFFSET)
    assert (np.abs(g(AT_OFFSET) - expected) < TOL)


# ---------------------------------------------------------------------------
# Simulation functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("simulation, expected_sum", [
    # On the DTLZ1 front the objectives sum to 0.5
    (dtlz1_sim, 0.5),
])
def test_dtlz_simulation_sums_to(simulation, expected_sum):
    """ Check the DTLZ simulations whose outputs sum to a known value. """

    sim = simulation(SIM_DTYPE, num_obj=NUM_OBJ, offset=OFFSET)
    assert (sum(sim(AT_OFFSET)) - expected_sum < TOL)


@pytest.mark.parametrize("simulation", [
    # DTLZ2 through DTLZ6 all have a unit-sphere front
    dtlz2_sim,
    dtlz3_sim,
    dtlz4_sim,
    dtlz5_sim,
    dtlz6_sim,
])
def test_dtlz_simulation_on_unit_sphere(simulation):
    """ Check the DTLZ simulations whose front is the unit sphere. """

    sim = simulation(SIM_DTYPE, num_obj=NUM_OBJ, offset=OFFSET)
    assert (np.linalg.norm(sim(AT_OFFSET)) - 1.0 < TOL)


@pytest.mark.parametrize("simulation", [dtlz8_sim, dtlz9_sim])
def test_dtlz_constrained_simulation_at_offset(simulation):
    """ Check the constrained DTLZ simulations at the kernel minimizer. """

    sim = simulation(SIM_DTYPE, num_obj=NUM_OBJ, offset=OFFSET)
    assert (np.all(np.abs(sim(AT_OFFSET)) < TOL))


def test_dtlz7_simulation():
    """ Check the DTLZ7 simulation, whose last output is discontinuous.

    DTLZ7's final objective is built from the others, so it is checked at a
    point with the first two design variables zeroed rather than at the
    kernel minimizer.

    """

    sim = dtlz7_sim(SIM_DTYPE, num_obj=NUM_OBJ, offset=OFFSET)
    x_in = dict(AT_OFFSET, x1=0.0, x2=0.0)
    assert (np.abs(sim(x_in)[2] - 6.0) < TOL)


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("objective, expected", [
    # DTLZ1's objectives are products of design variables, so at the origin
    # only the last one is nonzero
    (dtlz1_obj, [0.0, 0.0, 0.5]),
    # DTLZ2-4 use trigonometric terms, so at the origin the first is one
    (dtlz2_obj, [1.0, 0.0, 0.0]),
    (dtlz3_obj, [1.0, 0.0, 0.0]),
    (dtlz4_obj, [1.0, 0.0, 0.0]),
])
def test_dtlz_objectives_at_origin(objective, expected):
    """ Check each DTLZ objective at the origin of the design space. """

    x = np.zeros(1, dtype=OBJ_XTYPE)[0]
    sx = np.zeros(1, dtype=OBJ_STYPE)[0]
    for index, want in enumerate(expected):
        obj = objective(OBJ_XTYPE, OBJ_STYPE, index, num_obj=NUM_OBJ)
        assert (np.abs(obj(x, sx) - want) < 1.0e-8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

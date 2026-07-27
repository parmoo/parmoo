""" Unit tests for MOOP_base design-variable embedding and simulation packing.

"""

import numpy as np
import pytest

from parmoo import MOOP
from parmoo.embeddings import IdentityEmbedder
from parmoo.optimizers import LocalSurrogate_PS
from parmoo.tests.unit_tests.helpers import sim_dict, sim_norm

# The unit-cube fractions to round-trip: the lower bounds, the upper bounds,
# and five random interior points.
UNIT_POINTS = np.concatenate(
    ([np.zeros(7), np.ones(7)], np.random.sample((5, 7))), axis=0
)


@pytest.fixture(scope="module")
def mixed_moop():
    """ A MOOP with one design variable of every supported type. """

    moop = MOOP(LocalSurrogate_PS)
    moop.addDesign({'des_type': "integer", 'lb': 0, 'ub': 1000})
    moop.addDesign({'des_type': "continuous", 'lb': -1.0, 'ub': 0.0})
    moop.addDesign({'des_type': "categorical", 'levels': 2})
    moop.addDesign({'des_type': "categorical",
                    'levels': ["0guy", "1guy", "2guy"]})
    moop.addDesign({'des_type': "custom", 'embedder': IdentityEmbedder,
                    'lb': 0.0, 'ub': 1.0})
    moop.addDesign({'des_type': "raw", 'lb': 0.0, 'ub': 5.0})
    moop.addDesign({'des_type': "continuous", 'lb': 0.0, 'ub': 1.0})
    return moop


def feature_point(unit):
    """ Map a vector of 7 unit-cube fractions into the feature space. """

    return {"x1": int(1000 * unit[0]),
            "x2": unit[1] - 1.0,
            "x3": np.round(unit[2]),
            "x4": f"{int(np.round(unit[3]))}guy",
            "x5": unit[4],
            "x6": unit[5] * 5.0,
            "x7": unit[6]}


@pytest.mark.parametrize("unit", UNIT_POINTS)
def test_embed_produces_legal_latent_point(mixed_moop, unit):
    """ Check that _embed() lands inside the latent unit cube.

    Every embedded coordinate except the "raw" variable, which is passed
    through unscaled, must lie in [0, 1].

    """

    xxi = mixed_moop._embed(feature_point(unit))
    assert (xxi.size == mixed_moop.n_latent)
    # x1 through x5 occupy latent coordinates 0-5 (x4 expands to three)
    assert (np.all(xxi[:6] >= -1.0e-8) and np.all(xxi[:6] <= 1 + 1.0e-8))
    # Coordinate 6 is the raw variable, which is deliberately unscaled
    assert (xxi[7] >= -1.0e-8 and xxi[7] <= 1 + 1.0e-8)


@pytest.mark.parametrize("unit", UNIT_POINTS)
def test_extract_inverts_embed(mixed_moop, unit):
    """ Check that _extract() recovers every design variable type exactly.

    Integers and categoricals must come back identical, and the continuous,
    custom, and raw variables must agree to the design tolerance.

    """

    xi = feature_point(unit)
    out = mixed_moop._extract(mixed_moop._embed(xi))
    # Integer and categorical variables round-trip exactly
    assert (out["x1"] == xi["x1"])
    assert (out["x4"] == xi["x4"])
    assert (out["x5"] == xi["x5"])
    # Continuous, categorical-as-number, and raw variables round-trip to tol
    for key in ["x2", "x3", "x6", "x7"]:
        assert (np.abs(out[key] - xi[key]) < 1.0e-8)


def test_pack_unpack_sim():
    """ Check that simulation outputs pack to and from a flat latent array.

    _pack_sim() flattens the named simulation outputs into the array the
    surrogates consume, and _unpack_sim() restores the names.

    """

    moop = MOOP(LocalSurrogate_PS)
    moop.addDesign({'name': "x1", 'lb': 0.0, 'ub': 1000.0},
                   {'name': "x2", 'lb': -1.0, 'ub': 0.0})
    moop.addSimulation(sim_dict(1, sim_norm), sim_dict(2, sim_norm))
    # Build the same value in both representations
    flat = np.array([1.0, 2.0, 3.0])
    named = np.zeros(1, dtype=moop.sim_schema)
    named[0]['sim1'] = 1.0
    named[0]['sim2'][:] = np.array([2.0, 3.0])
    # Packing the named form yields the flat form
    assert (np.all(moop._pack_sim(named) == flat))
    # Unpacking the flat form yields the named form
    unpacked = moop._unpack_sim(flat)
    assert (unpacked['sim1'] == named[0]['sim1'])
    assert (np.all(unpacked['sim2'] == named[0]['sim2']))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

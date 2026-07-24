""" Unit tests for MOOP_base design-variable embedding and simulation packing.
"""


import pytest


def test_MOOP_base_embed_extract():
    """ Test that the MOOP_base class can embed/extract design variables.

    Add several design variables and generate an embedding. Then embed and
    extract several inputs, and check that the results match up to the
    design tolerance.

    """

    from jax import config
    config.update("jax_enable_x64", True)
    import numpy as np
    from parmoo import MOOP
    from parmoo.embeddings import IdentityEmbedder
    from parmoo.optimizers import LocalSurrogate_PS

    # Create a MOOP with 6 variables of mixed types
    moop = MOOP(LocalSurrogate_PS)
    moop.addDesign({'des_type': "integer",
                    'lb': 0,
                    'ub': 1000})
    moop.addDesign({'des_type': "continuous",
                    'lb': -1.0,
                    'ub': 0.0})
    moop.addDesign({'des_type': "categorical",
                    'levels': 2})
    moop.addDesign({'des_type': "categorical",
                    'levels': ["0guy", "1guy", "2guy"]})
    moop.addDesign({'des_type': "custom",
                    'embedder': IdentityEmbedder,
                    'lb': 0.0, 'ub': 1.0})
    moop.addDesign({'des_type': "raw",
                    'lb': 0.0, 'ub': 5.0})
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    # Embed/extract lower bounds, upper bounds, and 5 random values
    test_pts = np.zeros((7, 7))
    test_pts[1, :] = 1.0
    test_pts[2:, :] = np.random.sample((5, 7))
    for xi_tmp in test_pts:
        xi_tmp = np.random.sample(7)
        xi = {}
        xi["x1"] = int(1000 * xi_tmp[0])
        xi["x2"] = xi_tmp[1] - 1.0
        xi["x3"] = np.round(xi_tmp[2])
        xi["x4"] = f"{int(np.round(xi_tmp[3]))}guy"
        xi["x5"] = xi_tmp[4]
        xi["x6"] = xi_tmp[5] * 5.0
        xi["x7"] = xi_tmp[6]
        xxi = moop._embed(xi)
        # Check that embedding is legal
        assert (np.all(xxi[:6] >= -1.0e-8) and np.all(xxi[:6] <= 1 + 1.0e-8))
        assert (xxi[7] >= -1.0e-8 and xxi[7] <= 1 + 1.0e-8)
        assert (xxi.size == moop.n_latent)
        # Check extraction matches
        assert (moop._extract(xxi)["x1"] == xi["x1"])
        assert (np.abs(moop._extract(xxi)["x2"] - xi["x2"]) < 1.0e-8)
        assert (np.abs(moop._extract(xxi)["x3"] - xi["x3"]) < 1.0e-8)
        assert (moop._extract(xxi)["x4"] == xi["x4"])
        assert (moop._extract(xxi)["x5"] == xi["x5"])
        assert (np.abs(moop._extract(xxi)["x6"] - xi["x6"]) < 1.0e-8)
        assert (np.abs(moop._extract(xxi)["x7"] - xi["x7"]) < 1.0e-8)


def test_MOOP_base_pack_unpack_sim():
    """ Check that the MOOP_base class handles simulation packing correctly.

    Initialize a MOOP_base object with and without design variable names.
    Add 2 simulations and pack/unpack each output.

    """

    import numpy as np
    from parmoo import MOOP
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF

    # Create a continuous MOOP with 2 sims for packing/unpacking
    moop = MOOP(LocalSurrogate_PS)
    moop.addDesign({'name': "x1", 'lb': 0.0, 'ub': 1000.0},
                   {'name': "x2", 'lb': -1.0, 'ub': 0.0})
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    g2 = {'m': 2,
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x-1.0), np.linalg.norm(x-0.5)],
          'surrogate': GaussRBF}
    moop.addSimulation(g1, g2)
    # Create a test vector
    sx = np.array([1.0, 2.0, 3.0])
    sxx = np.zeros(1, dtype=moop.sim_schema)
    sxx[0]['sim1'] = 1.0
    sxx[0]['sim2'][:] = np.array([2.0, 3.0])
    # Check packing
    assert (np.all(moop._pack_sim(sxx) == sx))
    # Check unpacking
    assert (moop._unpack_sim(sx)['sim1'] == sxx[0]['sim1'])
    assert (moop._unpack_sim(sx)['sim2'][0] == sxx[0]['sim2'][0])
    assert (moop._unpack_sim(sx)['sim2'][1] == sxx[0]['sim2'][1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

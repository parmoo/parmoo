""" Unit tests for MOOP checkpointing: save, load, and reload fidelity.
"""

import pytest


def test_MOOP_save_load_functions():
    """ Check that a MOOP object can be correctly saved/reloaded.

    Create and save a MOOP object, then reload and check that it is the same.

    """

    import numpy as np
    import os
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    import pytest

    # Functions sim1, sim2, f1, f2, c1 need to be global for save/load to work
    global sim1, sim2, f1, f2, c1
    def sim1(x): return [np.sqrt(sum([x[i]**2 for i in x]))]
    def sim2(x): return [np.sqrt(sum([(x[i] - 1)**2 for i in x]))]
    def f1(x, sim): return sim["sim1"]
    def f2(x, sim): return sim["sim2"]
    def c1(x, sim): return x["x1"] - 0.5
    # Create a MOOP with 3 variables, 2 sims, 2 objs, and 1 constraint
    moop1 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100})
    # Empty save
    moop1.save()
    # Add MOOP variables, sims, objectives, etc.
    for i in range(2):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addDesign({'des_type': "categorical", 'levels': 3})
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 20,
          'sim_func': sim1,
          'surrogate': GaussRBF}
    g2 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 25,
          'sim_func': sim2,
          'surrogate': GaussRBF}
    moop1.addSimulation(g1, g2)
    # Add 2 objectives
    moop1.addObjective({'obj_func': f1},
                       {'obj_func': f2})
    # Add 1 constraint
    moop1.addConstraint({'constraint': c1})
    # Add 3 acquisition functions
    for i in range(3):
        moop1.addAcquisition({'acquisition': UniformWeights})
    # Activate data checkpointing
    moop1.database.setCheckpoint(True)
    # Collect some data
    batch = moop1.iterate(0)
    batch = moop1.filterBatch(batch)
    for (xi, i) in batch:
        moop1.evaluateSimulation(xi, i)
    moop1.updateAll(0, batch)
    # Test save
    moop1.save()
    # Test load
    moop2 = MOOP(LocalSurrogate_PS)
    moop2.load()
    # Check that save/load are correct
    check_moops(moop1, moop2)
    # Create a new MOOP with same specs
    moop3 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100})
    for i in range(2):
        moop3.addDesign({'lb': 0.0, 'ub': 1.0})
    moop3.addDesign({'des_type': "categorical", 'levels': 3})
    moop3.addSimulation(g1, g2)
    moop3.addObjective({'obj_func': f1},
                       {'obj_func': f2})
    moop3.addConstraint({'constraint': c1})
    for i in range(3):
        moop3.addAcquisition({'acquisition': UniformWeights})
    moop3.compile()
    # Try to save and overwrite old data
    with pytest.raises(OSError):
        moop3.save()
    # Start a checkpoint file with moop1
    moop1.setCheckpoint(True)
    # Try to overwrite with moop3
    with pytest.raises(OSError):
        moop3.setCheckpoint(True)
    # Clean up test directory
    os.remove("parmoo.moop")
    os.remove("parmoo.simdb.json")
    os.remove("parmoo.surrogate.1")
    os.remove("parmoo.surrogate.2")
    os.remove("parmoo.optimizer")


def test_MOOP_save_load_classes():
    """ Check that a MOOP object can be correctly saved/reloaded.

    Create and save a MOOP object, then reload and check that it is the same.

    Use simulation/objective callable objects from the library.

    """

    import os
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.constraints import SingleSimBound, SingleSimBoundGradient
    from parmoo.objectives import SingleSimObjective, SingleSimGradient
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.simulations.dtlz import dtlz2_sim

    # Create a mixed-variable MOOP with 3 variables, 2 sims, 2 objs, 1 const
    moop1 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100})
    # Test empty save
    moop1.save()
    for i in range(2):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addDesign({'des_type': "categorical", 'levels': 3})
    # Initialize the simulation group with 3 outputs
    g1 = {'m': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 20,
          'sim_func': dtlz2_sim(moop1.getDesignType(), num_obj=2),
          'surrogate': GaussRBF}
    moop1.addSimulation(g1)
    # Define and add the objectives and constraints
    f1 = SingleSimObjective(moop1.getDesignType(),
                            moop1.getSimulationType(),
                            ("sim1", 0))
    f2 = SingleSimObjective(moop1.getDesignType(),
                            moop1.getSimulationType(),
                            ("sim1", 1))
    df1 = SingleSimGradient(moop1.getDesignType(),
                            moop1.getSimulationType(),
                            ("sim1", 0))
    df2 = SingleSimGradient(moop1.getDesignType(),
                            moop1.getSimulationType(),
                            ("sim1", 1))
    c1 = SingleSimBound(moop1.getDesignType(),
                        moop1.getSimulationType(),
                        ("sim1", 1))
    dc1 = SingleSimBoundGradient(moop1.getDesignType(),
                                 moop1.getSimulationType(),
                                 ("sim1", 1))
    moop1.addObjective({'obj_func': f1, 'obj_grad': df1},
                       {'obj_func': f2, 'obj_grad': df2})
    moop1.addConstraint({'con_func': c1, 'con_grad': dc1})
    for i in range(3):
        moop1.addAcquisition({'acquisition': UniformWeights})
    moop1.compile()
    moop1.setCheckpoint(True)
    # Test save and reload
    moop1.save()
    moop2 = MOOP(LocalSurrogate_PS)
    moop2.load()
    check_moops(moop1, moop2)
    # Clean up test directory
    os.remove("parmoo.moop")
    os.remove("parmoo.simdb.json")
    os.remove("parmoo.surrogate.1")
    os.remove("parmoo.optimizer")


def test_MOOP_checkpoint():
    """ Check that the MOOP object performs checkpointing correctly.

    Run 1 iteration of ParMOO, with checkpointing on.

    """

    import numpy as np
    import os
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF

    # Functions sim1, sim2, f1, f2, c1 need to be global for save/load to work
    global sim1, sim2, f1, f2, c1
    def sim1(x): return [np.sqrt(sum([x[i] ** 2 for i in x]))]
    def sim2(x): return [np.sqrt(sum([(x[i] - 1) ** 2 for i in x]))]
    def f1(x, sim): return sim["sim1"]
    def f2(x, sim): return sim["sim2"]
    def c1(x, sim): return x["x1"] - 0.5
    # Create a mixed-variable MOOP with 3 variables, 2 sims, 3 objs, 1 const
    moop1 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100})
    for i in range(2):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addDesign({'des_type': "categorical", 'levels': 3})
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 20,
          'sim_func': sim1,
          'surrogate': GaussRBF}
    g2 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 25,
          'sim_func': sim2,
          'surrogate': GaussRBF}
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'obj_func': f1},
                       {'obj_func': f2})
    moop1.addConstraint({'constraint': c1})
    for i in range(3):
        moop1.addAcquisition({'acquisition': UniformWeights})
    # Turn on checkpointing
    moop1.setCheckpoint(True)
    # One iteration
    batch = moop1.iterate(0)
    batch = moop1.filterBatch(batch)
    for (xi, i) in batch:
        moop1.evaluateSimulation(xi, i)
    moop1.updateAll(0, batch)
    # Test load
    moop2 = MOOP(LocalSurrogate_PS)
    moop2.load()
    check_moops(moop1, moop2)
    # Clean up test directory
    os.remove("parmoo.moop")
    os.remove("parmoo.simdb.json")
    os.remove("parmoo.surrogate.1")
    os.remove("parmoo.surrogate.2")
    os.remove("parmoo.optimizer")


def check_moops(moop1, moop2):
    """ Auxiliary function for checking that 2 moops are equal.

    Check that all entries in moop1 = moop2

    Args:
        moop1 (MOOP): First moop to compare

        moop2 (MOOP): Second moop to compare

    """

    # Check scalars
    assert (moop2.m == moop1.m and
            moop2.n_feature == moop1.n_feature and
            moop2.n_latent == moop1.n_latent and
            moop2.o == moop1.o and moop2.p == moop1.p and
            moop2.s == moop1.s and
            len(moop2.getObjectiveData()) == len(moop2.getObjectiveData()) and
            moop2.penalty_param == moop1.penalty_param and
            moop2.iteration == moop1.iteration)
    # Check lists
    assert (all([dt2i == dt1i for dt2i, dt1i in zip(moop2.latent_des_tols,
                                                    moop1.latent_des_tols)]))
    assert (all([lb2i == lb1i for lb2i, lb1i in zip(moop2.latent_lb,
                                                    moop1.latent_lb)]))
    assert (all([ub2i == ub1i for ub2i, ub1i in zip(moop2.latent_ub,
                                                    moop1.latent_ub)]))
    assert (all([m2i == m1i for m2i, m1i in zip(moop2.m_list, moop1.m_list)]))
    assert (all([n2i[0] == n1i[0] for n2i, n1i in zip(moop2.sim_schema,
                                                      moop1.sim_schema)]))
    assert (all([n2i[0] == n1i[0] for n2i, n1i in zip(moop2.des_schema,
                                                      moop1.des_schema)]))
    assert (all([n2i[0] == n1i[0] for n2i, n1i in zip(moop2.obj_schema,
                                                      moop1.obj_schema)]))
    assert (all([n2i[0] == n1i[0] for n2i, n1i in zip(moop2.con_schema,
                                                      moop1.con_schema)]))
    # Check dictionaries
    assert all([
        x1 == x2
        for x1, x2 in zip(moop2.getObjectiveData(), moop1.getObjectiveData())
    ])
    assert all([
        x1 == x2
        for x1, x2 in zip(moop2.getSimulationData(), moop1.getSimulationData())
    ])
    for obj1, obj2 in zip(moop1.obj_funcs, moop2.obj_funcs):
        if hasattr(obj1, "__name__"):
            assert (obj1.__name__ == obj2.__name__)
        else:
            assert (obj1.__class__.__name__ == obj2.__class__.__name__)
    for sim1, sim2 in zip(moop1.sim_funcs, moop2.sim_funcs):
        if hasattr(sim1, "__name__"):
            assert (sim1.__name__ == sim2.__name__)
        else:
            assert (sim1.__class__.__name__ == sim2.__class__.__name__)
    for const1, const2 in zip(moop1.con_funcs, moop2.con_funcs):
        if hasattr(const1, "__name__"):
            assert (const1.__name__ == const2.__name__)
        else:
            assert (const1.__class__.__name__ == const2.__class__.__name__)
    # Check functions
    assert (moop2.optimizer.__class__.__name__ ==
            moop1.optimizer.__class__.__name__)
    assert (all([s1.__class__.__name__ == s2.__class__.__name__
                 for s1, s2 in zip(moop1.searches, moop2.searches)]))
    assert (all([s1.__class__.__name__ == s2.__class__.__name__
                 for s1, s2 in zip(moop1.surrogates, moop2.surrogates)]))
    assert (all([s1.__class__.__name__ == s2.__class__.__name__
                 for s1, s2 in zip(moop1.acquisitions, moop2.acquisitions)]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

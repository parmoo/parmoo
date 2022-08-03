def test_parallel_coordinates_static():

    from parmoo.viz.plot import parallel_coordinates
    import pytest
    import os
    parallel_coordinates(run_quickstart(), output='png')
    parallel_coordinates(run_quickstart(), output='png')
    parallel_coordinates(run_quickstart(), output='png')
    parallel_coordinates(run_quickstart(), output='png')
    assert(os.path.exists("Pareto front.png"))
    os.remove("Pareto front.png")


def run_quickstart():

    import numpy as np
    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalGPS

    my_moop = MOOP(LocalGPS)

    my_moop.addDesign({
        'name': "x1",
        'des_type': "continuous",
        'lb': 0.0, 'ub': 1.0
    })
    my_moop.addDesign({
        'name': "x2",
        'des_type': "categorical",
        'levels': 3
    })

    def sim_func(x):
        if x["x2"] == 0:
            return np.array([(x["x1"] - 0.2) ** 2, (x["x1"] - 0.8) ** 2])
        else:
            return np.array([99.9, 99.9])

    my_moop.addSimulation({
        'name': "MySim",
        'm': 2,
        'sim_func': sim_func,
        'search': LatinHypercube,
        'surrogate': GaussRBF,
        'hyperparams': {'search_budget': 20}
    })

    my_moop.addObjective({
        'name': "Cost of driving",
        'obj_func': lambda x, s: s["MySim"][0]
    })
    my_moop.addObjective({
        'name': "Time spent driving",
        'obj_func': lambda x, s: s["MySim"][1]
    })

    my_moop.addConstraint({
        'name': "c1",
        'constraint': lambda x, s: 0.1 - x["x1"]
    })

    for i in range(3):
        my_moop.addAcquisition({
            'acquisition': UniformWeights,
            'hyperparams': {}
        })

    my_moop.solve(5)

    return my_moop


def run_dtlz2():
    from parmoo import MOOP
    from parmoo.acquisitions import RandomConstraint
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.optimizers import LBFGSB
    from parmoo.objectives.dtlz import dtlz2_obj
    from parmoo.simulations.dtlz import g2_sim

    n = 6  # number of design variables
    o = 5  # number of objectives
    q = 4  # batch size (number of acquisitions)

    # Create MOOP
    moop = MOOP(LBFGSB)
    # Add n design variables
    for i in range(n):
        moop.addDesign({
            'name': f"Input {i+1}",
            'des_type': 'continuous',
            'lb': 0.0,
            'ub': 1.0,
            'des_tol': 1.0e-8
        })

    # Create the g2 simulation
    moop.addSimulation({
        'name': "g2",
        'm': 1,
        'sim_func': g2_sim(
            moop.getDesignType(),
            num_obj=o,
            offset=0.5
        ),
        'search': LatinHypercube,
        'surrogate': GaussRBF,
        'hyperparams': {'search_budget': 10*n}
    })
    # Add o objectives
    for i in range(o):
        moop.addObjective({
            'name': f"Objective {i+1}",
            'obj_func': dtlz2_obj(
                moop.getDesignType(),
                moop.getSimulationType(),
                i, num_obj=o)
        })

    # Add q acquisition functions
    for i in range(q):
        moop.addAcquisition({'acquisition': RandomConstraint})
    # Solve the MOOP with 20 iterations
    moop.solve(5)

    return moop


def run_checkpointing():

    import numpy as np
    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalGPS
    import logging

    # Create a new MOOP
    my_moop = MOOP(LocalGPS)

    # Add 1 continuous and 1 categorical design variable
    my_moop.addDesign({
        'name': "x1",
        'des_type': "continuous",
        'lb': 0.0,
        'ub': 1.0
    })
    my_moop.addDesign({
        'name': "x2",
        'des_type': "categorical",
        'levels': 3
    })

    # Create a simulation function
    def sim_func(x):
        if x["x2"] == 0:
            return np.array([(x["x1"] - 0.2) ** 2, (x["x1"] - 0.8) ** 2])
        else:
            return np.array([99.9, 99.9])

    # Add the simulation function to the MOOP
    my_moop.addSimulation({
        'name': "MySim",
        'm': 2,
        'sim_func': sim_func,
        'search': LatinHypercube,
        'surrogate': GaussRBF,
        'hyperparams': {'search_budget': 20}
    })

    # Define the 2 objectives as named Python functions
    def obj1(x, s): return s["MySim"][0]
    def obj2(x, s): return s["MySim"][1]

    # Define the constraint as a function
    def const(x, s): return 0.1 - x["x1"]

    # Add 2 objectives
    my_moop.addObjective({'name': "f1", 'obj_func': obj1})
    my_moop.addObjective({'name': "f2", 'obj_func': obj2})

    # Add 1 constraint
    my_moop.addConstraint({'name': "c1", 'constraint': const})

    # Add 3 acquisition functions (generates batches of size 3)
    for i in range(3):
        my_moop.addAcquisition({'acquisition': UniformWeights,
                                'hyperparams': {}})

    # Turn on logging with timestamps
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Use checkpointing without saving a separate data file
    # (in "parmoo.moop" file)
    my_moop.setCheckpoint(True, checkpoint_data=False, filename="parmoo")

    # Solve the problem with 4 iterations
    my_moop.solve(4)

    # Create a new MOOP object and reload the MOOP from parmoo.moop file
    new_moop = MOOP(LocalGPS)
    new_moop.load("parmoo")

    # Do another iteration
    new_moop.solve(5)

    return(new_moop)


def run_named():

    import numpy as np
    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalGPS

    my_moop = MOOP(LocalGPS)

    # Define a simulation to use below
    def sim_func(x):
        if x["MyCat"] == 0:
            return np.array([(x["MyDes"]) ** 2, (x["MyDes"] - 1.0) ** 2])
        else:
            return np.array([99.9, 99.9])

    # Add a design variable, simulation, objective, and constraint.
    # Note the 'name' keys for each
    my_moop.addDesign({
        'name': "MyDes",
        'des_type': "continuous",
        'lb': 0.0,
        'ub': 1.0
    })
    my_moop.addDesign({
        'name': "MyCat",
        'des_type': "categorical",
        'levels': 2
    })

    my_moop.addSimulation({
        'name': "MySim",
        'm': 2,
        'sim_func': sim_func,
        'search': LatinHypercube,
        'surrogate': GaussRBF,
        'hyperparams': {'search_budget': 20}
    })

    my_moop.addObjective({
        'name': "MyObj",
        'obj_func': lambda x, s: sum(s["MySim"])
    })

    my_moop.addConstraint({
        'name': "MyCon",
        'constraint': lambda x, s: 0.1 - x["MyDes"]
    })

    return my_moop

if __name__ == "__main__":
    test_parallel_coordinates_static()


import numpy as np
from parmoo import MOOP
from parmoo.acquisitions import RandomConstraint, FixedWeights
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.optimizers import GlobalSurrogate_BFGS

# Fix the random seed for reproducibility
np.random.seed(0)

# Create a new MOOP with a derivative-based solver
my_moop = MOOP(GlobalSurrogate_BFGS, hyperparams={})

# Add 3 continuous variables named x1, x2, x3
for i in range(3):
    my_moop.addDesign({'name': "x" + str(i+1),
                       'des_type': "continuous",
                       'lb': 0.0,
                       'ub': 1.0,
                       'des_tol': 1.0e-8})
# Add one categorical variable named x4
my_moop.addDesign({'name': "x4",
                   'des_type': "categorical",
                   'levels': 3})

def quad_sim(x):
    """ A quadratic simulation function with 2 outputs.

    Returns:
        np.ndarray: simulation value (S) with 2 outputs
         * S_1(x) = <x, x>
         * S_2(x) = <x-1, x-1>

    """

    return np.array([x["x1"] ** 2 + x["x2"] ** 2 + x["x3"] ** 2,
                     (x["x1"] - 1.0) ** 2 + (x["x2"] - 1.0) ** 2 +
                     (x["x3"] - 1.0) ** 2])

# Add the quadratic simulation to the problem
# Use a 10 point LH search for ex design and a Gaussian RBF surrogate model
my_moop.addSimulation({'name': "f_conv",
                       'm': 2,
                       'sim_func': quad_sim,
                       'search': LatinHypercube,
                       'surrogate': GaussRBF,
                       'hyperparams': {'search_budget': 10}})

def obj_f1(x, sim, der=0):
    """ Minimize the first output from 'f_conv' """

    if der == 0:
        return sim['f_conv'][0]
    elif der == 1:
        return np.zeros(1, dtype=x.dtype)[0]
    elif der == 2:
        result = np.zeros(1, dtype=sim.dtype)[0]
        result['f_conv'][0] = 1.0
        return result

def obj_f2(x, sim, der=0):
    """ Minimize the second output from 'f_conv' """

    if der == 0:
        return sim['f_conv'][1]
    elif der == 1:
        return np.zeros(1, dtype=x.dtype)[0]
    elif der == 2:
        result = np.zeros(1, dtype=sim.dtype)[0]
        result['f_conv'][1] = 1.0
        return result

# Minimize each of the 2 outputs from the quadratic simulation
my_moop.addObjective({'name': "f1",
                      'obj_func': obj_f1})
my_moop.addObjective({'name': "f2",
                      'obj_func': obj_f2})

def const_x4(x, sim, der=0):
    """ Constrain x["x4"] = 0 """

    if der == 0:
        return 0.0 if (x["x4"] == 0) else 1.0
    elif der == 1:
        # No derivatives for categorical design var, just return all zeros
        return np.zeros(1, dtype=x.dtype)[0]
    elif der == 2:
        return np.zeros(1, dtype=sim.dtype)[0]

# Add the single constraint to the problem
my_moop.addConstraint({'name': "c_x4",
                       'constraint': const_x4})

# Add 2 different acquisition functions to the problem
my_moop.addAcquisition({'acquisition': RandomConstraint})
my_moop.addAcquisition({'acquisition': FixedWeights,
                        # Fixed weight with equal weight on both objectives
                        'hyperparams': {'weights': np.array([0.5, 0.5])}})

# Turn on checkpointing -- creates the files parmoo.moop and parmoo.surrogate.1
my_moop.setCheckpoint(True, checkpoint_data=False, filename="parmoo")

# Turn on logging
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Solve the problem
my_moop.solve(5)

# Get and print full simulation database
sim_db = my_moop.getSimulationData(format="pandas")
print("Simulation data:")
for key in sim_db.keys():
    print(f"\t{key}:")
    print(sim_db[key])

# Get and print results
soln = my_moop.getPF(format="pandas")
print("\n\n")
print("Solution points:")
print(soln)

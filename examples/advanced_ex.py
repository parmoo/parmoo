
import numpy as np
from parmoo import MOOP
from parmoo.acquisitions import RandomConstraint, FixedWeights
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.optimizers import GlobalSurrogate_BFGS

# Create a new MOOP with a derivative-based solver
my_moop = MOOP(GlobalSurrogate_BFGS,
               # Use the hyperparams to fix the random seed for reproducibility
               hyperparams={'np_random_gen': 0})

# Add 3 continuous variables named x1, x2, x3
for i in range(3):
    my_moop.addDesign({'name': f"x{i+1}",
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

# Define some objectives below -- try to avoid things that jax can't compile

def obj_f1_func(x, sim):
    """ Minimize the first output from 'f_conv' """

    return sim['f_conv'][0]

def obj_f1_grad(x, sim):
    """ Corresponding gradient evaluations for obj_f1_func """

    dx = {'x1': 0.0, 'x2': 0.0, 'x3': 0.0, 'x4': 0.0}
    ds = {'f_conv': np.eye(2)[0]}
    return dx, ds

def obj_f2_func(x, sim):
    """ Minimize the second output from 'f_conv' """

    return sim['f_conv'][1]

def obj_f2_grad(x, sim):
    """ Corresponding gradient evaluations for obj_f2_func """

    dx = {'x1': 0.0, 'x2': 0.0, 'x3': 0.0, 'x4': 0.0}
    ds = {'f_conv': np.eye(2)[1]}
    return dx, ds

# Minimize each of the 2 outputs from the quadratic simulation
my_moop.addObjective({'name': "f1",
                      'obj_func': obj_f1_func,
                      'obj_grad': obj_f1_grad})
my_moop.addObjective({'name': "f2",
                      'obj_func': obj_f2_func,
                      'obj_grad': obj_f2_grad})

def const_x4_func(x, sim):
    """ Constrain x["x4"] = 0 """

    return 1.0 - (x["x4"] == 0)

def const_x4_grad(x, sim):
    """ Gradient for evaluating whether x["x4"] = 0 """

    # Note: There is no partial derivative for a categorical design variable.
    # This may make it hard to solve problems that place constraints on many
    # categorical variables, but for 1 categorical variable it should be okay.
    # We can just set all gradients equal to 0.
    dx = {'x1': 0.0, 'x2': 0.0, 'x3': 0.0, 'x4': 0.0}
    ds = {'f_conv': np.zeros(2)}
    return dx, ds

# Add the single constraint to the problem
my_moop.addConstraint({'name': "c_x4",
                       'con_func': const_x4_func,
                       'con_grad': const_x4_grad})

# Add 2 different acquisition functions to the problem
my_moop.addAcquisition({'acquisition': RandomConstraint})
my_moop.addAcquisition({'acquisition': FixedWeights,
                        # Fixed weights with equal weight on both objectives
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

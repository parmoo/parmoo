from parmoo import MOOP
from parmoo.optimizers import TR_LBFGSB
from parmoo.surrogates import LocalGaussRBF
from parmoo.acquisitions import RandomConstraint
from parmoo.searches import LatinHypercube
from parmoo.simulations.dtlz import g2_sim as sim_func
from parmoo.objectives.dtlz import dtlz2_obj as obj_func
from parmoo.constraints import single_sim_bound as const_func
import numpy as np

# Set the problem dimensions
NUM_DES = 3
NUM_OBJ = 3

# Create a MOOP
moop = MOOP(TR_LBFGSB)

# Add NUM_DES continuous design variables
for i in range(NUM_DES):
    moop.addDesign({'name': f"x{i+1}", 'ub': 1.0, 'lb': 0.0,
                    'des_type': "continuous", 'des_tol': 1.0e-8})

# Add the simulation
moop.addSimulation({'name': "g2",
                    'm': 1,
                    'sim_func': sim_func(moop.getDesignType(),
                                         num_obj=NUM_OBJ,
                                         offset=0.6),
                    'search': LatinHypercube,
                    'surrogate': LocalGaussRBF,
                    'hyperparams': {}})

# Add NUM_OBJ objective functions
for i in range(NUM_OBJ):
    moop.addObjective({'name': f"DTLZ2 objective {i+1}",
                       'obj_func': obj_func(moop.getDesignType(),
                                            moop.getSimulationType(),
                                            obj_ind=i, num_obj=NUM_OBJ)})

# Add a constraint
moop.addConstraint({'name': "Max Sim Bounds",
                    'constraint': const_func(moop.getDesignType(),
                                             moop.getSimulationType(),
                                             sim_ind="g2",
                                             type="upper",
                                             bound=2.0)})

# Add 10 acquisition funcitons
for i in range(10):
    moop.addAcquisition({'acquisition': RandomConstraint, 'hyperparams': {}})

# Solve the problem with 5 iterations
moop.solve(5)

# Check that 150 simulations were evaluated and solutions are feasible
assert(moop.getObjectiveData().shape[0] == 150)
assert(moop.getSimulationData()['g2'].shape[0] == 150)
assert(all([sum([fi[f"DTLZ2 objective {i+1}"]**2 for i in range(NUM_OBJ)]) <= 4
            for fi in moop.getPF()]))

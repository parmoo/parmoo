""" This module contains a library of common objective functions, matching
ParMOO's interface.

The common objectives are:
 * ``min_sim`` -- minimize a single simulation output
 * ``max_sim`` -- minimize -1 * a single simulation output
 * ``min_sos`` -- minimize the sum-of-squares of several simulation outputs
 * ``max_sos`` -- minimize -1 * the sum-of-squares of several sim outs
 * ``min_sum`` -- minimize the sum of several simulation outputs
 * ``max_sum`` -- minimize -1 * the sum of several sim outs

"""

from parmoo.objectives import obj_func
import numpy as np


class min_sim(obj_func):
    """ Class for minimizing a single simulation output.

    Minimize a single simulation output as an objective:
    obj_func(x, sx, der=0) = sx[i].

    Also comes with built-in derivative support.

    Contains 2 methods:
     * ``__init__(des, sim, obj_ind)``
     * ``__call__(x, sim, der=0)``

    The ``__init__`` method inherits from the obj_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ1 problem.

    """

    def __init__(self, des, sim, obj_ind, num_obj=3):
        """ Constructor for DTLZ1 class.

        Args:
            des (list, tuple, or int): Either the numpy.dtype of the
                design variables (list or tuple) or the number of design
                variables (assumed to all be continuous, unnamed).

            sim (list, tuple, or int): Either the numpy.dtype of the
                simulation outputs (list or tuple) or the number of simulation
                outputs (assumed to all be unnamed).

            obj_ind (int): The index of the DTLZ1 objective to return.

            num_obj (int, optional): The number of objectives for this problem.
                Note that this effects the calculation of the objective value,
                but still only a single objective output is created per
                instance of this class. To add all objectives, create
                num_obj instances with obj_ind = 0, ..., num_obj - 1.

        """

        super().__init__(des, sim)
        if self.m != 1:
            raise ValueError("DTLZ1 only supports 1 simulation output, " +
                             "but " + str(self.m) + " were given")
        if not isinstance(obj_ind, int):
            raise TypeError("optional input obj_ind must have the int type")
        if obj_ind < 0:
            raise ValueError("obj_ind cannot be negative")
        self.obj_ind = obj_ind
        if not isinstance(num_obj, int):
            raise TypeError("optional input num_obj must have the int type")
        if num_obj < 0:
            raise ValueError("num_obj cannot be negative")
        self.o = num_obj
        return

    def __call__(self, x, sim, der=0):
        """ Define simulation evaluation.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

            sim (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the corresponding simulation
                outputs.

            der (int, optional): Specifies whether to take derivative
                (and wrt which variables).
                 * der=1: take derivatives wrt x
                 * der=2: take derivatives wrt sim
                 * other: no derivatives
                Default value is der=0.

        Returns:
            numpy.ndarray: The output of this simulation for the input x.

        """

        # Extract x into xx, if names are used
        xx = np.zeros(self.n)
        if self.use_names:
            for i, name in enumerate(self.des_type):
                xx[i] = x[name[0]]
        else:
            xx[:] = x[:]
        # Extract sim into sx, if names are used
        sx = np.zeros(self.m)
        if self.use_names:
            for i, name in enumerate(self.sim_type):
                sx[i] = sim[name[0]]
        else:
            for i, si in enumerate(sim):
                sx[i] = si
        # Evaluate derivative wrt xx
        if der == 1:
            dx = np.zeros(self.n)
            i = self.obj_ind
            for j in range(self.o - i):
                if j < self.o - i - 1:
                    dx[j] = (np.prod(xx[0:j]) * np.prod(xx[j+1:self.o-i])
                             * (1 + sx[0]) / 2)
                    if i > 0:
                        dx[j] *= (1.0 - xx[self.o - i - 1])
                elif j == self.o - i - 1 and i != 0:
                    dx[j] = -(np.prod(xx[0:self.o-i-1]) * (1 + sx[0]) / 2)
            if self.use_names:
                result = np.zeros(1, dtype=self.des_type)
                for i, name in enumerate(self.des_type):
                    result[0][name] = dx[i]
                return result[0]
            else:
                return dx
        # Evaluate derivative wrt sx
        elif der == 2:
            # Initialize output array
            ds = np.zeros(self.m)
            i = self.obj_ind
            ds[0] = np.prod(xx[:self.o - i - 1]) / 2
            if i > 0:
                ds[0] *= (1 - xx[self.o - i - 1])
            if self.use_names:
                result = np.zeros(1, dtype=self.sim_type)
                result[0][sim_type[0][0]] = ds[0]
                return result[0]
            else:
                return ds
        # Evaluate fx
        else:
            # Initialize output array
            fx = 1.0 + sx[0]
            # Calculate the output array
            i = self.obj_ind
            fx = np.prod(xx[:self.o - 1 - i]) * (1 + sx[0]) / 2
            if i > 0:
                fx *= (1 - xx[self.o - 1 - i])
            return fx

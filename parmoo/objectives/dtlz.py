""" This module contains objective function implementations of the DTLZ test
suite, as described in:

Deb, Thiele, Laumanns, and Zitzler. "Scalable test problems for
evolutionary multiobjective optimization" in Evolutionary Multiobjective
Optimization, Theoretical Advances and Applications, Ch. 6 (pp. 105--145).
Springer-Verlag, London, UK, 2005. Abraham, Jain, and Goldberg (Eds).

Since DTLZ[1-7] depended upon kernel functions (implemented in
parmoo.simulations.dtlz), each of these problems is implemented here
as an algebraic, differentiable objective, with the kernel function output
as an input. The problems DTLZ8 and DTLZ9 do not support this modification,
so they are omitted.

TODO: DTLZ5, DTLZ6, and DTLZ7 have not yet been added.

To use this module, first import one or more of the following simulation/kernel
functions from parmoo.simulations.dtlz:
 * ``g1_sim``
 * ``g2_sim``
 * ``g3_sim``
 * ``g4_sim``

The 7 DTLZ problems included here are:
 * ``dtlz1_obj``
 * ``dtlz2_obj``
 * ``dtlz3_obj``
 * ``dtlz4_obj``

"""

from parmoo.objectives import obj_func
from parmoo.util import to_array, from_array
import numpy as np


class dtlz1_obj(obj_func):
    """ Class defining the DTLZ1 objectives.

    Use this class in combination with the g1_sim() class from the
    parmoo.simulations.dtlz module

    DTLZ1 has a linear Pareto front, with all nondominated points
    on the hyperplane F_1 + F_2 + ... + F_o = 0.5.
    DTLZ1 has 11^k - 1 "local" Pareto fronts where k = n - m + 1, and
    1 "global" Pareto front.

    Contains 2 methods:
     * ``__init__(des, sim, obj_ind)``
     * ``__call__(x, sim, der=0)``

    The ``__init__`` method inherits from the obj_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ1 problem.

    """

    def __init__(self, des, sim, obj_ind, num_obj=3):
        """ Constructor for DTLZ1 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

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

    def __call__(self, x, sx):
        """ Define simulation evaluation.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

            sx (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the corresponding simulation
                outputs.

        Returns:
            numpy.ndarray: The output of this simulation for the input x.

        """

        # Roll x into xx and sx into ssx
        xx = to_array(x, self.des_type)
        ssx = to_array(sx, self.sim_type)
        # Evaluate derivative wrt xx
        if False:
            dx = np.zeros(self.n)
            i = self.obj_ind
            for j in range(self.o - i):
                if j < self.o - i - 1:
                    dx[j] = (np.prod(xx[0:j]) * np.prod(xx[j+1:self.o-i-1])
                             * (1 + ssx[0]) / 2)
                    if i > 0:
                        dx[j] *= (1.0 - xx[self.o - i - 1])
                elif j == self.o - i - 1 and i != 0:
                    dx[j] = -(np.prod(xx[0:self.o-i-1]) * (1 + ssx[0]) / 2)
            i = self.obj_ind
            ds = np.array(np.prod(xx[:self.o - i - 1]) / 2).flatten()
            if i > 0:
                ds *= (1 - xx[self.o - i - 1])
            return from_array(dx, self.des_type), from_array(ds, self.sim_type)
        # Initialize output array
        fx = 1.0 + ssx[0]
        i = self.obj_ind
        fx = jnp.prod(xx[:self.o - 1 - i]) * (1 + ssx[0]) / 2
        if i > 0:
            fx *= (1 - xx[self.o - 1 - i])
        return fx


class dtlz2_obj(obj_func):
    """ Class defining the DTLZ2 objectives.

    Use this class in combination with the g2_sim() class from the
    parmoo.simulations.dtlz module.

    DTLZ2 has a concave Pareto front, given by the unit sphere in
    objective space, restricted to the positive orthant.
    DTLZ2 has no "local" Pareto fronts, besides the true Pareto front.

    Contains 2 methods:
     * ``__init__(des, sim, obj_ind)``
     * ``__call__(x, sim, der=0)``

    The ``__init__`` method inherits from the obj_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ2 problem.

    """

    def __init__(self, des, sim, obj_ind, num_obj=3):
        """ Constructor for DTLZ2 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            sim (list, tuple, or int): Either the numpy.dtype of the
                simulation outputs (list or tuple) or the number of simulation
                outputs (assumed to all be unnamed).

            obj_ind (int): The index of the DTLZ2 objective to return.

            num_obj (int, optional): The number of objectives for this problem.
                Note that this effects the calculation of the objective value,
                but still only a single objective output is created per
                instance of this class. To add all objectives, create
                num_obj instances with obj_ind = 0, ..., num_obj - 1.

        """

        super().__init__(des, sim)
        if self.m != 1:
            raise ValueError("DTLZ2 only supports 1 simulation output, " +
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

    def __call__(self, x, s):
        """ Define simulation evaluation.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

            sx (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the corresponding simulation
                outputs.

        Returns:
            numpy.ndarray: The output of this simulation for the input x.

        """

        # Extract x into xx and sim into sx, if names are used
        xx = unpack(x, self.des_type)
        ssx = unpack(sx, self.sim_type)
        # Evaluate derivative wrt xx
        if False:
            dx = np.zeros(self.n)
            i = self.obj_ind
            for j in range(self.o - i):
                if j < self.o - i - 1:
                    dx[j] = (np.prod(np.cos(xx[:j] * np.pi / 2)) *
                             (-np.pi / 2) * np.sin(xx[j] * np.pi / 2) *
                             np.prod(np.cos(xx[j+1:self.o-i-1] * np.pi / 2)) *
                             (1 + ssx[0]))
                    if i > 0:
                        dx[j] *= np.sin(xx[self.o - i - 1] * np.pi / 2)
                elif j == self.o - i - 1 and i != 0:
                    dx[j] = (np.prod(np.cos(xx[0:self.o-i-1] * np.pi / 2)) *
                             (1 + ssx[0]) * (np.pi / 2) *
                             np.cos(xx[self.o - i - 1] * np.pi / 2))
            i = self.obj_ind
            ds = np.array(np.prod(np.cos(xx[:self.o - i - 1] * np.pi / 2))).flatten()
            if i > 0:
                ds *= np.sin(np.pi * xx[self.o - i - 1] / 2)
            return from_array(dx, self.des_type), from_array(ds, self.sim_type)
        # Evaluate fx
        fx = 1.0 + ssx[0]
        i = self.obj_ind
        fx *= jnp.prod(jnp.cos(xx[:self.o - i - 1] * jnp.pi / 2))
        if i > 0:
            fx *= jnp.sin(jnp.pi * xx[self.o - i - 1] / 2)
        return fx


class dtlz3_obj(obj_func):
    """ Class defining the DTLZ3 objectives.

    Use this class in combination with the g1_sim() class from the
    parmoo.simulations.dtlz module.

    DTLZ3 has a concave Pareto front, given by the unit sphere in
    objective space, restricted to the positive orthant.
    DTLZ3 has 3^k - 1 "local" Pareto fronts where k = n - o + 1, and
    1 "global" Pareto front.

    Contains 2 methods:
     * ``__init__(des, sim, obj_ind)``
     * ``__call__(x, sim, der=0)``

    The ``__init__`` method inherits from the obj_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ3 problem.

    """

    def __init__(self, des, sim, obj_ind, num_obj=3):
        """ Constructor for DTLZ3 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            sim (list, tuple, or int): Either the numpy.dtype of the
                simulation outputs (list or tuple) or the number of simulation
                outputs (assumed to all be unnamed).

            obj_ind (int): The index of the DTLZ3 objective to return.

            num_obj (int, optional): The number of objectives for this problem.
                Note that this effects the calculation of the objective value,
                but still only a single objective output is created per
                instance of this class. To add all objectives, create
                num_obj instances with obj_ind = 0, ..., num_obj - 1.

        """

        super().__init__(des, sim)
        if self.m != 1:
            raise ValueError("DTLZ3 only supports 1 simulation output, " +
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

        # Extract x into xx and sim into sx, if names are used
        xx = unpack(x, self.des_type)
        sx = unpack(sim, self.sim_type)
        # Evaluate derivative wrt xx
        if der == 1:
            dx = np.zeros(self.n)
            i = self.obj_ind
            for j in range(self.o - i):
                if j < self.o - i - 1:
                    dx[j] = (np.prod(np.cos(xx[:j] * np.pi / 2)) *
                             (-np.pi / 2) * np.sin(xx[j] * np.pi / 2) *
                             np.prod(np.cos(xx[j+1:self.o-i-1] * np.pi / 2)) *
                             (1 + sx[0]))
                    if i > 0:
                        dx[j] *= np.sin(xx[self.o - i - 1] * np.pi / 2)
                elif j == self.o - i - 1 and i != 0:
                    dx[j] = (np.prod(np.cos(xx[0:self.o-i-1] * np.pi / 2)) *
                             (1 + sx[0]) * (np.pi / 2) *
                             np.cos(xx[self.o - i - 1] * np.pi / 2))
            if self.use_names:
                result = np.zeros(1, dtype=self.des_type)
                for i, name in enumerate(self.des_type.names):
                    result[0][name] = dx[i]
                return result[0]
            else:
                return dx
        # Evaluate derivative wrt sx
        elif der == 2:
            # Initialize output array
            ds = np.zeros(self.m)
            i = self.obj_ind
            ds[0] = np.prod(np.cos(xx[:self.o - i - 1] * np.pi / 2))
            if i > 0:
                ds[0] *= np.sin(np.pi * xx[self.o - i - 1] / 2)
            if self.use_names:
                result = np.zeros(1, dtype=self.sim_type)
                for name in self.sim_type.names:
                    result[0][name] = ds[0]
                return result[0]
            else:
                return ds
        # Evaluate fx
        else:
            # Initialize output array
            fx = 1.0 + sx[0]
            # Calculate the output array
            i = self.obj_ind
            fx *= np.prod(np.cos(xx[:self.o - i - 1] * np.pi / 2))
            if i > 0:
                fx *= np.sin(np.pi * xx[self.o - i - 1] / 2)
            return fx


class dtlz4_obj(obj_func):
    """ Class defining the DTLZ4 objectives.

    Use this class in combination with the g2_sim() class from the
    parmoo.simulations.dtlz module.

    DTLZ4 has a concave Pareto front, given by the unit sphere in
    objective space, restricted to the positive orthant.
    DTLZ4 has no "local" Pareto fronts, besides the true Pareto front,
    but by tuning the optional parameter alpha, one can adjust the
    solution density, making it harder for MOO algorithms to produce
    a uniform distribution of solutions.

    Contains 2 methods:
     * ``__init__(des, sim, obj_ind)``
     * ``__call__(x, sim, der=0)``

    The ``__init__`` method inherits from the obj_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ4 problem.

    """

    def __init__(self, des, sim, obj_ind, num_obj=3, alpha=100.0):
        """ Constructor for DTLZ4 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            sim (list, tuple, or int): Either the numpy.dtype of the
                simulation outputs (list or tuple) or the number of simulation
                outputs (assumed to all be unnamed).

            obj_ind (int): The index of the DTLZ4 objective to return.

            num_obj (int, optional): The number of objectives for this problem.
                Note that this effects the calculation of the objective value,
                but still only a single objective output is created per
                instance of this class. To add all objectives, create
                num_obj instances with obj_ind = 0, ..., num_obj - 1.

            alpha (optional, float or int): The uniformity parameter used for
                controlling the uniformity of the distribution of solutions
                across the Pareto front. Must be greater than or equal to 1.
                A value of 1 results in DTLZ2. Default value is 100.0.

        """

        super().__init__(des, sim)
        if self.m != 1:
            raise ValueError("DTLZ4 only supports 1 simulation output, " +
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
        if not (isinstance(alpha, int) or isinstance(alpha, float)):
            raise TypeError("alpha must be a numeric type")
        if alpha < 1:
            raise ValueError("alpha must be at least 1")
        self.alpha = alpha
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

        # Extract x into xx and sim into sx, if names are used
        xx = unpack(x, self.des_type)
        sx = unpack(sim, self.sim_type)
        # Evaluate derivative wrt xx
        if der == 1:
            dx = np.zeros(self.n)
            i = self.obj_ind
            for j in range(self.o - i):
                if j < self.o - i - 1:
                    dx[j] = (np.prod(np.cos(xx[:j] ** self.alpha * np.pi / 2))
                             * (-np.pi * self.alpha / 2)
                             * xx[j] ** (self.alpha - 1)
                             * np.sin(xx[j] ** self.alpha * np.pi / 2) *
                             np.prod(np.cos(xx[j+1:self.o-i-1] ** self.alpha *
                                            np.pi / 2)) * (1 + sx[0]))
                    if i > 0:
                        dx[j] *= np.sin(xx[self.o - i - 1] ** self.alpha *
                                        np.pi / 2)
                elif j == self.o - i - 1 and i != 0:
                    dx[j] = (np.prod(np.cos(xx[0:self.o-i-1] ** self.alpha *
                                            np.pi / 2)) * (1 + sx[0]) *
                             (np.pi * self.alpha / 2) *
                             xx[j] ** (self.alpha - 1) *
                             np.cos(xx[j] ** self.alpha * np.pi / 2))
            if self.use_names:
                result = np.zeros(1, dtype=self.des_type)
                for i, name in enumerate(self.des_type.names):
                    result[0][name] = dx[i]
                return result[0]
            else:
                return dx
        # Evaluate derivative wrt sx
        elif der == 2:
            # Initialize output array
            ds = np.zeros(self.m)
            i = self.obj_ind
            ds[0] = np.prod(np.cos(xx[:self.o - i - 1] ** self.alpha *
                                   np.pi / 2))
            if i > 0:
                ds[0] *= np.sin(xx[self.o - i - 1] ** self.alpha * np.pi / 2)
            if self.use_names:
                result = np.zeros(1, dtype=self.sim_type)
                for name in self.sim_type.names:
                    result[0][name] = ds[0]
                return result[0]
            else:
                return ds
        # Evaluate fx
        else:
            # Initialize output array
            fx = 1.0 + sx[0]
            # Calculate the output array
            i = self.obj_ind
            fx *= np.prod(np.cos(xx[:self.o - i - 1] ** self.alpha *
                                 np.pi / 2))
            if i > 0:
                fx *= np.sin(xx[self.o - i - 1] ** self.alpha * np.pi / 2)
            return fx

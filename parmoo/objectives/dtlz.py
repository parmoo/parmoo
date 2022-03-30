""" This module contains objective function implementations of the DTLZ test
suite, as described in:

Deb, Thiele, Laumanns, and Zitzler. "Scalable test problems for
evolutionary multiobjective optimization" in Evolutionary Multiobjective
Optimization, Theoretical Advances and Applications, Ch. 6 (pp. 105--145).
Springer-Verlag, London, UK, 2005. Abraham, Jain, and Goldberg (Eds).

Since DTLZ1-7 depended upon kernel functions (implemented in
parmoo.simulations.dtlz), each of these problems is implemented here
as an algebraic, differentiable objective, with the kernel function output
as an input. The problems DTLZ8 and DTLZ9 do not support this modification,
so they are omitted.

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
 * ``dtlz5_obj``
 * ``dtlz6_obj``
 * ``dtlz7_obj``

"""

from parmoo.simulations import sim_func
import numpy as np


class dtlz1_obj(obj_func):
    """ Class defining the DTLZ1 objectives.

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

    def __init__(self, des, sim, obj_ind):
        """ Constructor for DTLZ1 class.

        Args:
            des (list, tuple, or int): Either the numpy.dtype of the
                design variables (list or tuple) or the number of design
                variables (assumed to all be continuous, unnamed).

            sim (list, tuple, or int): Either the numpy.dtype of the
                simulation outputs (list or tuple) or the number of simulation
                outputs (assumed to all be unnamed).

            obj_ind (int): The index of the DTLZ1 objective to return.

        """

        super().__init__(self, des, sim)
        if not isinstance(obj_ind, int):
            raise TypeError("optional input obj_ind must have the int type")
        if obj_ind < 0:
            raise ValueError("obj_ind cannot be negative")
        self.obj_ind = obj_ind
        return

    def __call__(self, x, sim, der=0):
        """ Define simulation evaluation.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

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
            for j in range(self.n):
                if j < self.o - i + 1:
                    dx[j] = (np.prod(xx[0:j]) * np.prod(xx[j+1:self.o-i])
                             * (1 + sx) / 2)
                    if i > 0:
                        dx[j] *= (1.0 - xx[self.o - i + 1]) 
                elif j == self.o - i + 1 and i != 0:
                    dx[j] = -(np.prod(xx[0:self.o-i]) * (1 + sx) / 2)
            if use_names:
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
            ds = np.prod(xx[:self.o - 1 - i]) / 2
            if i > 0:
                ds[0] *= (1 - xx[self.o - 1 - i])
            if use_names:
                result = np.zeros(1, dtype=self.sim_type)
                result[0][sim_type[0][0]] = ds[0]
                return result[0]
            else:
                return ds
        # Evaluate fx
        else:
            # Initialize output array
            fx = 1.0 + sx
            # Calculate the output array
            i = self.obj_ind
            fx = np.prod(xx[:self.o - 1 - i]) * (1 + sx) / 2
            if i > 0:
                fx *= (1 - xx[self.o - 1 - i])
            return fx


class dtlz2_sim(sim_func):
    """ Class defining the DTLZ2 problem with offset minimizer.

    DTLZ2 has a concave Pareto front, given by the unit sphere in
    objective space, restricted to the positive orthant.
    DTLZ2 has no "local" Pareto fronts, besides the true Pareto front.

    Contains 2 methods:
     * ``__init__(des, sim)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ2 problem.

    """

    def __init__(self, des, sim, offset=0.5):
        """ Constructor for DTLZ2 class.

        Args:
            des (list, tuple, or int): Either the numpy.dtype of the
                design variables (list or tuple) or the number of design
                variables (assumed to all be continuous, unnamed).

            sim (list, tuple, or int): Either the numpy.dtype of the
                simulation outputs (list or tuple) or the number of simulation
                outputs (assumed to all be unnamed).

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.5.

        """

        super().__init__(self, des, sim)
        if not isinstance(offset, float):
            raise TypeError("optional input offset must have the float type")
        if offset < 0 or offset > 1:
            raise ValueError("offset must be in the range [0, 1]")
        self.offset = offset
        return

    def __call__(self, x):
        """ Define simulation evaluation.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

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
        # Initialize kernel function
        ker = g2_sim(self.n, self.m, self.offset)
        # Initialize output array
        fx = np.zeros(self.m)
        fx[:] = (1.0 + ker(xx)[0])
        # Calculate the output array
        for i in range(self.m - 1):
            for j in range(self.m - 1 - i):
                fx[i] *= np.cos(np.pi * xx[j] / 2)
            if i > 0:
                fx[i] *= np.sin(np.pi * xx[self.m - 1 - i] / 2)
        return fx


class dtlz3_sim(sim_func):
    """ Class defining the DTLZ3 problem with offset minimizer.

    DTLZ3 has a concave Pareto front, given by the unit sphere in
    objective space, restricted to the positive orthant.
    DTLZ3 has 3^k - 1 "local" Pareto fronts where k = n - m + 1, and
    1 "global" Pareto front.

    Contains 2 methods:
     * ``__init__(des, sim)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ3 problem.

    """

    def __init__(self, des, sim, offset=0.5):
        """ Constructor for DTLZ3 class.

        Args:
            des (list, tuple, or int): Either the numpy.dtype of the
                design variables (list or tuple) or the number of design
                variables (assumed to all be continuous, unnamed).

            sim (list, tuple, or int): Either the numpy.dtype of the
                simulation outputs (list or tuple) or the number of simulation
                outputs (assumed to all be unnamed).

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.5.

        """

        super().__init__(self, des, sim)
        if not isinstance(offset, float):
            raise TypeError("optional input offset must have the float type")
        if offset < 0 or offset > 1:
            raise ValueError("offset must be in the range [0, 1]")
        self.offset = offset
        return

    def __call__(self, x):
        """ Define simulation evaluation.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

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
        # Initialize kernel function
        ker = g1_sim(self.n, self.m, self.offset)
        # Initialize output array
        fx = np.zeros(self.m)
        fx[:] = (1.0 + ker(xx)[0])
        # Calculate the output array
        for i in range(self.m - 1):
            for j in range(self.m - 1 - i):
                fx[i] *= np.cos(np.pi * xx[j] / 2)
            if i > 0:
                fx[i] *= np.sin(np.pi * xx[self.m - 1 - i] / 2)
        return fx


class dtlz4_sim(sim_func):
    """ Class defining the DTLZ4 problem with offset minimizer.

    DTLZ4 has a concave Pareto front, given by the unit sphere in
    objective space, restricted to the positive orthant.
    DTLZ4 has no "local" Pareto fronts, besides the true Pareto front,
    but by tuning the optional parameter alpha, one can adjust the
    solution density, making it harder for MOO algorithms to produce
    a uniform distribution of solutions.

    Contains 2 methods:
     * ``__init__(des, sim)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ4 problem.

    """

    def __init__(self, des, sim, offset=0.5, alpha=100.0):
        """ Constructor for DTLZ4 class.

        Args:
            des (list, tuple, or int): Either the numpy.dtype of the
                design variables (list or tuple) or the number of design
                variables (assumed to all be continuous, unnamed).

            sim (list, tuple, or int): Either the numpy.dtype of the
                simulation outputs (list or tuple) or the number of simulation
                outputs (assumed to all be unnamed).

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.5.

            alpha (optional, float or int): The uniformity parameter used for
                controlling the uniformity of the distribution of solutions
                across the Pareto front. Must be greater than or equal to 1.
                A value of 1 results in DTLZ2. Default value is 100.0.

        """

        super().__init__(self, des, sim)
        if not isinstance(offset, float):
            raise TypeError("optional input offset must have the float type")
        if offset < 0 or offset > 1:
            raise ValueError("offset must be in the range [0, 1]")
        self.offset = offset
        if not (isinstance(alpha, float) or isinstance(alpha, int)):
            raise TypeError("optional input alpha must have a numeric type")
        if alpha < 1:
            raise ValueError("alpha must be at least 1")
        self.alpha = alpha
        return

    def __call__(self, x):
        """ Define simulation evaluation.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

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
        # Initialize kernel function
        ker = g2_sim(self.n, self.m, self.offset)
        # Initialize output array
        fx = np.zeros(self.m)
        fx[:] = (1.0 + ker(xx)[0])
        # Calculate the output array
        for i in range(self.m - 1):
            for j in range(self.m - 1 - i):
                fx[i] *= np.cos(np.pi * xx[j] ** self.alpha / 2)
            if i > 0:
                fx[i] *= np.sin(np.pi * xx[self.m - 1 - i] ** self.alpha / 2)
        return fx


class dtlz5_sim(sim_func):
    """ Class defining the DTLZ5 problem with offset minimizer.

    DTLZ5 has a lower-dimensional Pareto front embedded in the objective
    space, given by an arc of the unit sphere in the positive orthant.
    DTLZ5 has no "local" Pareto fronts, besides the true Pareto front.

    Contains 2 methods:
     * ``__init__(des, sim)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ5 problem.

    """

    def __init__(self, des, sim, offset=0.5):
        """ Constructor for DTLZ5 class.

        Args:
            des (list, tuple, or int): Either the numpy.dtype of the
                design variables (list or tuple) or the number of design
                variables (assumed to all be continuous, unnamed).

            sim (list, tuple, or int): Either the numpy.dtype of the
                simulation outputs (list or tuple) or the number of simulation
                outputs (assumed to all be unnamed).

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.5.

        """

        super().__init__(self, des, sim)
        if not isinstance(offset, float):
            raise TypeError("optional input offset must have the float type")
        if offset < 0 or offset > 1:
            raise ValueError("offset must be in the range [0, 1]")
        self.offset = offset
        return

    def __call__(self, x):
        """ Define simulation evaluation.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

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
        # Initialize kernel function
        ker = g2_sim(self.n, self.m, self.offset)
        # Calculate theta values
        theta = np.zeros(self.m - 1)
        g2x = ker(xx)
        for i in range(self.m - 1):
            theta[i] = np.pi * (1 + 2 * g2x * xx[i]) / (4 * (1 + g2x))
        # Initialize output array
        fx = np.zeros(self.m)
        fx[:] = (1.0 + g2x)
        # Calculate the output array
        for i in range(self.m - 1):
            for j in range(self.m - 1 - i):
                fx[i] *= np.cos(np.pi * theta[j] / 2)
            if i > 0:
                fx[i] *= np.sin(np.pi * theta[self.m - 1 - i] / 2)
        return fx


class dtlz6_sim(sim_func):
    """ Class defining the DTLZ6 problem with offset minimizer.

    DTLZ6 has a lower-dimensional Pareto front embedded in the objective
    space, given by an arc of the unit sphere in the positive orthant.
    DTLZ6 has no "local" Pareto fronts, but tends to show very little
    improvement until the algorithm is very close to its solution set.

    Contains 2 methods:
     * ``__init__(des, sim)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ6 problem.

    """

    def __init__(self, des, sim, offset=0.0):
        """ Constructor for DTLZ6 class.

        Args:
            des (list, tuple, or int): Either the numpy.dtype of the
                design variables (list or tuple) or the number of design
                variables (assumed to all be continuous, unnamed).

            sim (list, tuple, or int): Either the numpy.dtype of the
                simulation outputs (list or tuple) or the number of simulation
                outputs (assumed to all be unnamed).

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.0.

        """

        super().__init__(self, des, sim)
        if not isinstance(offset, float):
            raise TypeError("optional input offset must have the float type")
        if offset < 0 or offset > 1:
            raise ValueError("offset must be in the range [0, 1]")
        self.offset = offset
        return

    def __call__(self, x):
        """ Define simulation evaluation.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

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
        # Initialize kernel function
        ker = g3_sim(self.n, self.m, self.offset)
        # Calculate theta values
        theta = np.zeros(self.m - 1)
        g3x = ker(xx)
        for i in range(self.m - 1):
            theta[i] = np.pi * (1 + 2 * g3x * xx[i]) / (4 * (1 + g3x))
        # Initialize output array
        fx = np.zeros(self.m)
        fx[:] = (1.0 + g3x)
        # Calculate the output array
        for i in range(self.m - 1):
            for j in range(self.m - 1 - i):
                fx[i] *= np.cos(np.pi * theta[j] / 2)
            if i > 0:
                fx[i] *= np.sin(np.pi * theta[self.m - 1 - i] / 2)
        return fx


class dtlz7_sim(sim_func):
    """ Class defining the DTLZ7 problem with offset minimizer.

    DTLZ7 has a discontinuous Pareto front, with solutions on the 
    2^(m-1) discontinuous nondominated regions of the surface:

    F_m = m - F_1 (1 + sin(3pi F_1)) - ... - F_{m-1} (1 + sin3pi F_{m-1}).

    Contains 2 methods:
     * ``__init__(des, sim)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ7 problem.

    """

    def __init__(self, des, sim, offset=0.0):
        """ Constructor for DTLZ7 class.

        Args:
            des (list, tuple, or int): Either the numpy.dtype of the
                design variables (list or tuple) or the number of design
                variables (assumed to all be continuous, unnamed).

            sim (list, tuple, or int): Either the numpy.dtype of the
                simulation outputs (list or tuple) or the number of simulation
                outputs (assumed to all be unnamed).

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.0.

        """

        super().__init__(self, des, sim)
        if not isinstance(offset, float):
            raise TypeError("optional input offset must have the float type")
        if offset < 0 or offset > 1:
            raise ValueError("offset must be in the range [0, 1]")
        self.offset = offset
        return

    def __call__(self, x):
        """ Define simulation evaluation.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

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
        # Initialize kernel function
        ker = g4_sim(self.n, self.m, self.offset)
        # Initialize first m-1 entries in the output array
        fx = np.zeros(self.m)
        fx[:self.m-1] = xx[:self.m-1]
        # Calculate kernel functions
        gx = 1.0 + ker(xx)
        hx = (-np.sum(xx[:self.m-1] *
                      (1.0 + np.sin(3.0 * np.pi * xx[:self.m-1]) / gx))
                      + float(self.m))
        # Calculate the last entry in the output array
        fx[m] = gx * hx
        return fx


class dtlz8_sim(sim_func):
    """ Class defining the DTLZ8 problem with offset minimizer.

    DTLZ8 is a constrained MOOP, whose Pareto front combines a region
    of a plane with a line segment. To fully define DTLZ8, you must also
    use the parmoo.constraints.dtlz module.

    Contains 2 methods:
     * ``__init__(des, sim)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ8 problem.

    """

    def __init__(self, des, sim, offset=0.0):
        """ Constructor for DTLZ8 class.

        Args:
            des (list, tuple, or int): Either the numpy.dtype of the
                design variables (list or tuple) or the number of design
                variables (assumed to all be continuous, unnamed).

            sim (list, tuple, or int): Either the numpy.dtype of the
                simulation outputs (list or tuple) or the number of simulation
                outputs (assumed to all be unnamed).

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.0.

        """

        super().__init__(self, des, sim)
        if not isinstance(offset, float):
            raise TypeError("optional input offset must have the float type")
        if offset < 0 or offset > 1:
            raise ValueError("offset must be in the range [0, 1]")
        self.offset = offset
        return

    def __call__(self, x):
        """ Define simulation evaluation.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

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
        # Initialize the output array
        fx = np.zeros(self.m)
        # Calculate outputs
        for i in range(self.m):
            start = i * self.n // self.m
            stop = (i + 1) * self.n // self.m
            fx[i] = np.sum(np.abs(xx[start:stop] - self.offset))
        return fx


class dtlz9_sim(sim_func):
    """ Class defining the DTLZ9 problem with offset minimizer.

    DTLZ9 is a constrained MOOP, whose Pareto front is a subregion of
    the arc traced out by the solution to DTLZ5. To fully define DTLZ8,
    you must also use the parmoo.constraints.dtlz module.

    Contains 2 methods:
     * ``__init__(des, sim)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ9 problem.

    """

    def __init__(self, des, sim, offset=0.0):
        """ Constructor for DTLZ9 class.

        Args:
            des (list, tuple, or int): Either the numpy.dtype of the
                design variables (list or tuple) or the number of design
                variables (assumed to all be continuous, unnamed).

            sim (list, tuple, or int): Either the numpy.dtype of the
                simulation outputs (list or tuple) or the number of simulation
                outputs (assumed to all be unnamed).

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.0.

        """

        super().__init__(self, des, sim)
        if not isinstance(offset, float):
            raise TypeError("optional input offset must have the float type")
        if offset < 0 or offset > 1:
            raise ValueError("offset must be in the range [0, 1]")
        self.offset = offset
        return

    def __call__(self, x):
        """ Define simulation evaluation.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

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
        # Initialize the output array
        fx = np.zeros(self.m)
        # Calculate outputs
        for i in range(self.m):
            start = i * self.n // self.m
            stop = (i + 1) * self.n // self.m
            fx[i] = np.sum(np.abs(xx[start:stop] - self.offset) ** 0.1)
        return fx

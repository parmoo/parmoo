""" This module contains simulation function implementations of the DTLZ test
suite, as described in:

Deb, Thiele, Laumanns, and Zitzler. "Scalable test problems for
evolutionary multiobjective optimization" in Evolutionary Multiobjective
Optimization, Theoretical Advances and Applications, Ch. 6 (pp. 105--145).
Springer-Verlag, London, UK, 2005. Abraham, Jain, and Goldberg (Eds).

One drawback of the original DTLZ problems was that their global minima
(Pareto points) always corresponded to design points that satisfy

x_i = 0.5, for i = number of objectives, ..., number of design points

or

x_i = 0, for i = number of objectives, ..., number of design points.

This was appropriate for testing evolutionary algorithms, but for many
deterministic algorithms, these solutions may represent either the
best- or worst-case scenarios.

To make these problems applicable for deterministic algorithms, the
solution sets must be configurable offset by a user-specified amount,
as proposed in:

Chang. Mathematical Software for Multiobjective Optimization Problems.
Ph.D. dissertation, Virginia Tech, Dept. of Computer Science, 2020.

For the problems DTLZ8 and DTLZ9, only objective outputs are given
by the simulation function herein. To fully define the problem, also
use one or more of the corresponding constraint classes included in
``parmoo.constraints.dtlz`` [NOT YET IMPLEMENTED].

The full list of simulation functions in this module includes the kernel
functions:
 * ``g1_sim``
 * ``g2_sim``
 * ``g3_sim``
 * ``g4_sim``

and the 9 DTLZ problems in simulation form, with each simulation output
corresponding to an objective:
 * ``dtlz1_sim``
 * ``dtlz2_sim``
 * ``dtlz3_sim``
 * ``dtlz4_sim``
 * ``dtlz5_sim``
 * ``dtlz6_sim``
 * ``dtlz7_sim``
 * ``dtlz8_sim``
 * ``dtlz9_sim``

"""

from parmoo.simulations import sim_func
from parmoo.util import unpack
import numpy as np


class g1_sim(sim_func):
    """ Class defining 1 of 4 kernel functions used in the DTLZ problem suite.

    g1 = 100 ( (n - o + 1) +
               sum_{i=o}^n ((x_i - offset)^2 - cos(20pi(x_i - offset))) )

    Contains 2 methods:
     * ``__init__(des, num_obj)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the g1 kernel.

    """

    def __init__(self, des, num_obj=3, offset=0.5):
        """ Constructor for g1 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            num_obj (int, optional): The number of objectives, which is
                used to calculate the value of g1. Note that regardless of
                the number of objectives, the number of simulation
                outputs from g1 is always 1.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.5.

        """

        super().__init__(des)
        __check_optionals__(num_obj=num_obj, offset=offset)
        self.o = num_obj
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
        xx = unpack(x, self.des_type)
        # Calculate output
        result = (1 + self.n - self.o +
                  np.sum((xx[self.o-1:self.n] - self.offset) ** 2 -
                          np.cos(20.0 * np.pi *
                                 (xx[self.o-1:self.n] - self.offset)))) * 100.0
        return np.array([result])


class g2_sim(sim_func):
    """ Class defining 2 of 4 kernel functions used in the DTLZ problem suite.

    g2 = (x_o - offset)^2 + ... + (x_n - offset)^2

    Contains 2 methods:
     * ``__init__(des)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the g2 problem.

    """

    def __init__(self, des, num_obj=3, offset=0.5):
        """ Constructor for g2 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            num_obj (int, optional): The number of objectives, which is
                used to calculate the value of g2. Note that regardless of
                the number of objectives, the number of simulation
                outputs from g2 is always 1.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.5.

        """

        super().__init__(des)
        __check_optionals__(num_obj=num_obj, offset=offset)
        self.o = num_obj
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
        xx = unpack(x, self.des_type)
        return np.array([np.sum((xx[self.o-1:self.n] - self.offset) ** 2)])


class g3_sim(sim_func):
    """ Class defining 3 of 4 kernel functions used in the DTLZ problem suite.

    g3 = |x_o - offset|^.1 + ... + |x_n - offset|^.1

    Contains 2 methods:
     * ``__init__(des)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the g3 problem.

    """

    def __init__(self, des, num_obj=3, offset=0.0):
        """ Constructor for g3 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            num_obj (int, optional): The number of objectives, which is
                used to calculate the value of g3. Note that regardless of
                the number of objectives, the number of simulation
                outputs from g3 is always 1.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.0.

        """

        super().__init__(des)
        __check_optionals__(num_obj=num_obj, offset=offset)
        self.o = num_obj
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
        xx = unpack(x, self.des_type)
        return np.array([np.sum(np.abs(xx[self.o-1:self.n] - self.offset)
                                ** 0.1)])


class g4_sim(sim_func):
    """ Class defining 4 of 4 kernel functions used in the DTLZ problem suite.

    g4 = 1 + (9 * (|x_o - offset| + ... + |x_n - offset|) / (n + 1 - o))

    Contains 2 methods:
     * ``__init__(des)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the g4 problem.

    """

    def __init__(self, des, num_obj=3, offset=0.0):
        """ Constructor for g4 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            num_obj (int, optional): The number of objectives, which is
                used to calculate the value of g4. Note that regardless of
                the number of objectives, the number of simulation
                outputs from g4 is always 1.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.0.

        """

        super().__init__(des)
        __check_optionals__(num_obj=num_obj, offset=offset)
        self.o = num_obj
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
        xx = unpack(x, self.des_type)
        return np.array([(9 * np.sum(np.abs(xx[self.o-1:self.n] - self.offset))
                          / float(self.n + 1 - self.o)) + 1.0])


class dtlz1_sim(sim_func):
    """ Class defining the DTLZ1 problem with offset minimizer.

    DTLZ1 has a linear Pareto front, with all nondominated points
    on the hyperplane F_1 + F_2 + ... + F_o = 0.5.
    DTLZ1 has 11^k - 1 "local" Pareto fronts where k = n - o + 1, and
    1 "global" Pareto front.

    Contains 2 methods:
     * ``__init__(des)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ1 problem.

    """

    def __init__(self, des, num_obj=3, offset=0.5):
        """ Constructor for DTLZ1 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            num_obj (int, optional): The number of objectives, which is
                used as the number of simulation outputs.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.5.

        """

        super().__init__(des)
        __check_optionals__(num_obj=num_obj, offset=offset)
        self.o = num_obj
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
        xx = unpack(x, self.des_type)
        # Initialize kernel function
        ker = g1_sim(self.n, self.o, self.offset)
        # Initialize output array
        fx = np.zeros(self.o)
        fx[:] = (1.0 + ker(xx)[0]) / 2.0
        # Calculate the output array
        for i in range(self.o):
            for j in range(self.o - 1 - i):
                fx[i] *= xx[j]
            if i > 0:
                fx[i] *= (1.0 - xx[self.o - 1 - i])
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

    def __init__(self, des, num_obj=3, offset=0.5):
        """ Constructor for DTLZ2 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            num_obj (int, optional): The number of objectives, which is
                used as the number of simulation outputs.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.5.

        """

        super().__init__(des)
        __check_optionals__(num_obj=num_obj, offset=offset)
        self.o = num_obj
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
        xx = unpack(x, self.des_type)
        # Initialize kernel function
        ker = g2_sim(self.n, self.o, self.offset)
        # Initialize output array
        fx = np.zeros(self.o)
        fx[:] = (1.0 + ker(xx)[0])
        # Calculate the output array
        for i in range(self.o):
            for j in range(self.o - 1 - i):
                fx[i] *= np.cos(np.pi * xx[j] / 2)
            if i > 0:
                fx[i] *= np.sin(np.pi * xx[self.o - 1 - i] / 2)
        return fx


class dtlz3_sim(sim_func):
    """ Class defining the DTLZ3 problem with offset minimizer.

    DTLZ3 has a concave Pareto front, given by the unit sphere in
    objective space, restricted to the positive orthant.
    DTLZ3 has 3^k - 1 "local" Pareto fronts where k = n - o + 1, and
    1 "global" Pareto front.

    Contains 2 methods:
     * ``__init__(des, sim)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ3 problem.

    """

    def __init__(self, des, num_obj=3, offset=0.5):
        """ Constructor for DTLZ3 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            num_obj (int, optional): The number of objectives, which is
                used as the number of simulation outputs.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.5.

        """

        super().__init__(des)
        __check_optionals__(num_obj=num_obj, offset=offset)
        self.o = num_obj
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
        xx = unpack(x, self.des_type)
        # Initialize kernel function
        ker = g1_sim(self.n, self.o, self.offset)
        # Initialize output array
        fx = np.zeros(self.o)
        fx[:] = (1.0 + ker(xx)[0])
        # Calculate the output array
        for i in range(self.o):
            for j in range(self.o - 1 - i):
                fx[i] *= np.cos(np.pi * xx[j] / 2)
            if i > 0:
                fx[i] *= np.sin(np.pi * xx[self.o - 1 - i] / 2)
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

    def __init__(self, des, num_obj=3, offset=0.5, alpha=100.0):
        """ Constructor for DTLZ4 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            num_obj (int, optional): The number of objectives, which is
                used as the number of simulation outputs.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.5.

            alpha (optional, float or int): The uniformity parameter used for
                controlling the uniformity of the distribution of solutions
                across the Pareto front. Must be greater than or equal to 1.
                A value of 1 results in DTLZ2. Default value is 100.0.

        """

        super().__init__(des)
        __check_optionals__(num_obj=num_obj, offset=offset, alpha=alpha)
        self.o = num_obj
        self.offset = offset
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
        xx = unpack(x, self.des_type)
        # Initialize kernel function
        ker = g2_sim(self.n, self.o, self.offset)
        # Initialize output array
        fx = np.zeros(self.o)
        fx[:] = (1.0 + ker(xx)[0])
        # Calculate the output array
        for i in range(self.o):
            for j in range(self.o - 1 - i):
                fx[i] *= np.cos(np.pi * xx[j] ** self.alpha / 2)
            if i > 0:
                fx[i] *= np.sin(np.pi * xx[self.o - 1 - i] ** self.alpha / 2)
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

    def __init__(self, des, num_obj=3, offset=0.5):
        """ Constructor for DTLZ5 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            num_obj (int, optional): The number of objectives, which is
                used as the number of simulation outputs.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.5.

        """

        super().__init__(des)
        __check_optionals__(num_obj=num_obj, offset=offset)
        self.o = num_obj
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
        xx = unpack(x, self.des_type)
        # Initialize kernel function
        ker = g2_sim(self.n, self.o, self.offset)
        # Calculate theta values
        theta = np.zeros(self.o - 1)
        g2x = ker(xx)
        for i in range(self.o - 1):
            theta[i] = np.pi * (1 + 2 * g2x * xx[i]) / (4 * (1 + g2x))
        # Initialize output array
        fx = np.zeros(self.o)
        fx[:] = (1.0 + g2x)
        # Calculate the output array
        for i in range(self.o):
            for j in range(self.o - 1 - i):
                fx[i] *= np.cos(np.pi * theta[j] / 2)
            if i > 0:
                fx[i] *= np.sin(np.pi * theta[self.o - 1 - i] / 2)
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

    def __init__(self, des, num_obj=3, offset=0.0):
        """ Constructor for DTLZ6 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            num_obj (int, optional): The number of objectives, which is
                used as the number of simulation outputs.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.0.

        """

        super().__init__(des)
        __check_optionals__(num_obj=num_obj, offset=offset)
        self.o = num_obj
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
        xx = unpack(x, self.des_type)
        # Initialize kernel function
        ker = g3_sim(self.n, self.o, self.offset)
        # Calculate theta values
        theta = np.zeros(self.o - 1)
        g3x = ker(xx)
        for i in range(self.o - 1):
            theta[i] = np.pi * (1 + 2 * g3x * xx[i]) / (4 * (1 + g3x))
        # Initialize output array
        fx = np.zeros(self.o)
        fx[:] = (1.0 + g3x)
        # Calculate the output array
        for i in range(self.o):
            for j in range(self.o - 1 - i):
                fx[i] *= np.cos(np.pi * theta[j] / 2)
            if i > 0:
                fx[i] *= np.sin(np.pi * theta[self.o - 1 - i] / 2)
        return fx


class dtlz7_sim(sim_func):
    """ Class defining the DTLZ7 problem with offset minimizer.

    DTLZ7 has a discontinuous Pareto front, with solutions on the 
    2^(o-1) discontinuous nondominated regions of the surface:

    F_m = o - F_1 (1 + sin(3pi F_1)) - ... - F_{o-1} (1 + sin3pi F_{o-1}).

    Contains 2 methods:
     * ``__init__(des, sim)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ7 problem.

    """

    def __init__(self, des, num_obj=3, offset=0.0):
        """ Constructor for DTLZ7 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            num_obj (int, optional): The number of objectives, which is
                used as the number of simulation outputs.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.0.

        """

        super().__init__(des)
        __check_optionals__(num_obj=num_obj, offset=offset)
        self.o = num_obj
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
        xx = unpack(x, self.des_type)
        # Initialize kernel function
        ker = g4_sim(self.n, self.o, self.offset)
        # Initialize first o-1 entries in the output array
        fx = np.zeros(self.o)
        print(xx)
        fx[:self.o-1] = xx[:self.o-1]
        # Calculate kernel functions
        gx = 1.0 + ker(xx)
        hx = (-np.sum(xx[:self.o-1] *
                      (1.0 + np.sin(3.0 * np.pi * xx[:self.o-1]) / gx))
                      + float(self.o))
        # Calculate the last entry in the output array
        fx[self.o-1] = gx * hx
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

    def __init__(self, des, num_obj=3, offset=0.0):
        """ Constructor for DTLZ8 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            num_obj (int, optional): The number of objectives, which is
                used as the number of simulation outputs.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.0.

        """

        super().__init__(des)
        __check_optionals__(num_obj=num_obj, offset=offset)
        self.o = num_obj
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
        xx = unpack(x, self.des_type)
        # Initialize the output array
        fx = np.zeros(self.o)
        # Calculate outputs
        for i in range(self.o):
            start = i * self.n // self.o
            stop = (i + 1) * self.n // self.o
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

    def __init__(self, des, num_obj=3, offset=0.0):
        """ Constructor for DTLZ9 class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            num_obj (int, optional): The number of objectives, which is
                used as the number of simulation outputs.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = number of objectives, ..., number of
                design variables. The default value is offset = 0.0.

        """

        super().__init__(des)
        __check_optionals__(num_obj=num_obj, offset=offset)
        self.o = num_obj
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
        xx = unpack(x, self.des_type)
        # Initialize the output array
        fx = np.zeros(self.o)
        # Calculate outputs
        for i in range(self.o):
            start = i * self.n // self.o
            stop = (i + 1) * self.n // self.o
            fx[i] = np.sum(np.abs(xx[start:stop] - self.offset) ** 0.1)
        return fx


def __check_optionals__(num_obj=3, offset=0.5, alpha=100.0):
    """ Check DTLZ optional inputs for illegal values.

    Not recommended for external usage.

    """

    if not isinstance(offset, float):
        raise TypeError("optional input offset must have the float type")
    if offset < 0 or offset > 1:
        raise ValueError("offset must be in the range [0, 1]")
    if not isinstance(num_obj, int):
        raise TypeError("optional input num_obj must be an int type")
    if num_obj < 1:
        raise ValueError("optional input num_obj must be greater than 0")
    if not (isinstance(alpha, int) or isinstance(alpha, float)):
        raise TypeError("alpha must be a numeric type")
    if alpha < 1:
        raise ValueError("alpha must be at least 1")
    return

""" This module implements several common test problems, proposed in:

Deb, Thiele, Laumanns, and Zitzler. Scalable multi-objective optimization
test problems. In Proc. 2002 IEEE Congress on Evolutionary Computation,
pp. 825--830.

One drawback of the original DTLZ problems was that their global minima
(Pareto points) always corresponded to the values

x_i = 0.5, for i = number of objectives, ..., number of design points.

This was appropriate for testing evolutionary algorithms, but for many
deterministic sampling schemes, this would cause the Pareto front to be
immediately identified with an early sample.

To make these problems usable for deterministic algorithms, the minimizers
must be offset by some variable amount, as proposed in:

Chang. Mathematical Software for Multiobjective Optimization Problems.
Ph.D. dissertation, Virginia Tech, Dept. of Computer Science, 2020.

"""

from parmoo.sims.sim_func import sim_func


def g1(x, o, offset):
    """ 1 of 2 kernel functions used in the DTLZ problem suite.

    Args:
        x (np.ndarray): Input array specifying design point to evaluate.

        o (int): The number of objectives for this problem.

        offset (float): The location of the global minimizers for g1(x[o:]).

    Returns:
        float: Output g1 evaluation. See Deb et al. (2002) for more details.

    """

    import math

    return (float(len(x[o - 1:])) +
            sum((x[o-1:] - offset) ** 2 -
                math.cos(20.0 * math.pi * (x[o-1:] - offset)))) * 100.0


def g2(x, o, offset):
    """ 2 of 2 kernel functions used in the DTLZ problem suite.

    Args:
        x (np.ndarray): Input array specifying design point to evaluate.

        o (int): The number of objectives for this problem.

        offset (float): The location of the global minimizers for g2(x[o:]).

    Returns:
        float: Output g2 evaluation. See Deb et al. (2002) for more details.

    """

    return sum((x[o-1:] - offset) ** 2)


class dtlz1(sim_func):
    """ Class defining the DTLZ1 problem, originally proposed in:

    Deb, Thiele, Laumanns, and Zitzler. Scalable multi-objective optimization
    test problems. In Proc. 2002 IEEE Congress on Evolutionary Computation,
    pp. 825--830.

    Contains 2 methods:
     * ``__init__(des, sim)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the DTLZ1 problem.

    """

    def __init__(self, des, sim, offset=0.5):
        """ Constructor for DTLZ1 class.

        Args:
            des (list, tuple, or int): Either the numpy.dtype of the
                design variables (list or tuple) or the number of design
                variables (assumed to all be continuous, unnamed).

            sim (list, tuple, or int): Either the numpy.dtype of the
                simulation outputs (list or tuple) or the number of simulation
                outputs (assumed to all be unnamed).

        """

        super().__init__(self, des, sim)
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
        # Initialize output array
        fx = np.zeros(self.m)
        fx[:] = (1.0 + g1(xx)) / 2.0
        # Calculate the output array
        for i in range(self.o - 1):
            for j in range(self.o - 1 - i):
                fx[i] *= xx[j]
            if i > 0:
                fx[i] *= (1.0 - xx[self.o - 1 - i])
        return fx

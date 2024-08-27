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

from jax import numpy as jnp
import numpy as np
from parmoo.structs import CompositeFunction
from parmoo.util import to_array, from_array


class dtlz1_obj(CompositeFunction):
    """ Class defining the DTLZ1 objectives.

    Use this class in combination with the g1_sim() class from the
    parmoo.simulations.dtlz module

    DTLZ1 has a linear Pareto front, with all nondominated points
    on the hyperplane F_1 + F_2 + ... + F_o = 0.5.
    DTLZ1 has 11^k - 1 "local" Pareto fronts where k = n - m + 1, and
    1 "global" Pareto front.

    Contains 2 methods:
     * ``__init__(des_type, sim_type, obj_ind)``
     * ``__call__(x, sx)``

    The ``__init__`` method inherits from the CompositeFunction ABC.

    The ``__call__`` method performs an evaluation of the DTLZ1 problem.

    """

    def __init__(self, des_type, sim_type, obj_ind, num_obj=3):
        """ Constructor for DTLZ1 class.

        Args:
            des_type (np.dtype): The numpy.dtype of the design variables.

            sim_type (np.dtype): The numpy.dtype of the simulation outputs.

            obj_ind (int): The index of the DTLZ1 objective to return.

            num_obj (int, optional): The number of objectives for this problem.
                Note that this effects the calculation of the objective value,
                but still only a single objective output is created per
                instance of this class. To add all objectives, create
                num_obj instances with obj_ind = 0, ..., num_obj - 1.

        """

        super().__init__(des_type, sim_type)
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
        """ DTLZ1 objective function evaluation at the specified index.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            float: The output of this objective for the input x.

        """

        # Roll x into xx and sx into ssx
        xx = to_array(x, self.des_type)
        ssx = to_array(sx, self.sim_type)[0]
        # Evaluate
        i = self.obj_ind
        fx = (1 + ssx) * jnp.prod(xx[:self.o - 1 - i]) / 2
        if i > 0:
            fx *= (1 - xx[self.o - 1 - i])
        return fx


class dtlz2_obj(CompositeFunction):
    """ Class defining the DTLZ2 objectives.

    Use this class in combination with the g2_sim() class from the
    parmoo.simulations.dtlz module.

    DTLZ2 has a concave Pareto front, given by the unit sphere in
    objective space, restricted to the positive orthant.
    DTLZ2 has no "local" Pareto fronts, besides the true Pareto front.

    Contains 2 methods:
     * ``__init__(des_type, sim_type, obj_ind)``
     * ``__call__(x, sx)``

    The ``__init__`` method inherits from the CompositeFunction ABC.

    The ``__call__`` method performs an evaluation of the DTLZ2 problem.

    """

    def __init__(self, des_type, sim_type, obj_ind, num_obj=3):
        """ Constructor for DTLZ2 class.

        Args:
            des_type (np.dtype): The numpy.dtype of the design variables.

            sim_type (np.dtype): The numpy.dtype of the simulation outputs.

            obj_ind (int): The index of the DTLZ2 objective to return.

            num_obj (int, optional): The number of objectives for this problem.
                Note that this effects the calculation of the objective value,
                but still only a single objective output is created per
                instance of this class. To add all objectives, create
                num_obj instances with obj_ind = 0, ..., num_obj - 1.

        """

        super().__init__(des_type, sim_type)
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

    def __call__(self, x, sx):
        """ DTLZ2 objective function evaluation at the specified index.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            float: The output of this objective for the input x.

        """

        # Extract x into xx and sx into ssx
        xx = to_array(x, self.des_type)
        ssx = to_array(sx, self.sim_type)[0]
        # Evaluate
        i = self.obj_ind
        fx = (1 + ssx) * jnp.prod(jnp.cos(xx[:self.o - i - 1] * jnp.pi / 2))
        if i > 0:
            fx *= jnp.sin(jnp.pi * xx[self.o - i - 1] / 2)
        return fx


class dtlz3_obj(CompositeFunction):
    """ Class defining the DTLZ3 objectives.

    Use this class in combination with the g1_sim() class from the
    parmoo.simulations.dtlz module.

    DTLZ3 has a concave Pareto front, given by the unit sphere in
    objective space, restricted to the positive orthant.
    DTLZ3 has 3^k - 1 "local" Pareto fronts where k = n - o + 1, and
    1 "global" Pareto front.

    Contains 2 methods:
     * ``__init__(des_type, sim_type, obj_ind)``
     * ``__call__(x, sx)``

    The ``__init__`` method inherits from the CompositeFunction ABC.

    The ``__call__`` method performs an evaluation of the DTLZ3 problem.

    """

    def __init__(self, des_type, sim_type, obj_ind, num_obj=3):
        """ Constructor for DTLZ3 class.

        Args:
            des_type (np.dtype): The numpy.dtype of the design variables.

            sim_type (np.dtype): The numpy.dtype of the simulation outputs.

            obj_ind (int): The index of the DTLZ3 objective to return.

            num_obj (int, optional): The number of objectives for this problem.
                Note that this effects the calculation of the objective value,
                but still only a single objective output is created per
                instance of this class. To add all objectives, create
                num_obj instances with obj_ind = 0, ..., num_obj - 1.

        """

        super().__init__(des_type, sim_type)
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

    def __call__(self, x, sx):
        """ DTLZ3 objective function evaluation at the specified index.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            float: The output of this objective for the input x.

        """

        # Extract x into xx and sx into ssx
        xx = to_array(x, self.des_type)
        ssx = to_array(sx, self.sim_type)[0]
        # Evaluate
        i = self.obj_ind
        fx = (1 + ssx) * jnp.prod(jnp.cos(xx[:self.o - i - 1] * jnp.pi / 2))
        if i > 0:
            fx *= jnp.sin(jnp.pi * xx[self.o - i - 1] / 2)
        return fx


class dtlz4_obj(CompositeFunction):
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
     * ``__init__(des_type, sim_type, obj_ind)``
     * ``__call__(x, sx)``

    The ``__init__`` method inherits from the CompositeFunction ABC.

    The ``__call__`` method performs an evaluation of the DTLZ4 problem.

    """

    def __init__(self, des_type, sim_type, obj_ind, num_obj=3, alpha=100.0):
        """ Constructor for DTLZ4 class.

        Args:
            des_type (np.dtype): The numpy.dtype of the design variables.

            sim_type (np.dtype): The numpy.dtype of the simulation outputs.

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

        super().__init__(des_type, sim_type)
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

    def __call__(self, x, sx):
        """ DTLZ4 objective function evaluation at the specified index.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            float: The output of this objective for the input x.

        """

        # Extract x into xx and sx into ssx
        xx = to_array(x, self.des_type)
        ssx = to_array(sx, self.sim_type)[0]
        # Evaluate
        i = self.obj_ind
        fx = (1 + ssx) * np.prod(np.cos(xx[:self.o - i - 1] ** self.alpha *
                                 np.pi / 2))
        if i > 0:
            fx *= np.sin(xx[self.o - i - 1] ** self.alpha * np.pi / 2)
        return fx


class dtlz1_grad(dtlz1_obj):
    """ Class defining the DTLZ1 gradient.

    Inherits from ``dtlz1_obj``, but overwrites the ``__call__`` method to
    evaluate the gradient wrt x and sx, respectively.

    """

    def __call__(self, x, sx):
        """ DTLZ1 gradient evaluation.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            dict, dict: The gradient of this objective for the input (x, sx)
            wrt x and sx, respectively.

        """

        # Roll x into xx and sx into ssx
        xx = to_array(x, self.des_type)
        ssx = to_array(sx, self.sim_type)[0]
        # Evaluate grad wrt x
        i = self.obj_ind
        dx = jnp.zeros(self.n)
        for j in range(self.o - i - 1):
            dxj = (jnp.prod(xx[0:j]) * jnp.prod(xx[j + 1:self.o - i - 1])
                   * (1 + ssx) / 2)
            if i > 0:
                dxj *= (1.0 - xx[self.o - i - 1])
            dx = dx.at[j].set(dxj)
        if i != 0:
            j = self.o - i - 1
            dxj = -(jnp.prod(xx[:self.o - i - 1]) * (1 + ssx) / 2)
            dx = dx.at[j].set(dxj)
        # Evaluate grad wrt sx
        ds = jnp.prod(xx[:self.o - i - 1]) / 2
        if i > 0:
            ds *= (1 - xx[self.o - i - 1])
        return from_array(dx, self.des_type), from_array(ds, self.sim_type)


class dtlz2_grad(dtlz2_obj):
    """ Class defining the DTLZ2 gradient.

    Inherits from ``dtlz2_obj``, but overwrites the ``__call__`` method to
    evaluate the gradient wrt x and sx, respectively.

    """

    def __call__(self, x, sx):
        """ DTLZ2 gradient evaluation.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            dict, dict: The gradient of this objective for the input (x, sx)
            wrt x and sx, respectively.

        """

        # Extract x into xx and sx into ssx, if names are used
        xx = to_array(x, self.des_type)
        ssx = to_array(sx, self.sim_type)[0]
        # Evaluate derivative wrt xx
        i = self.obj_ind
        dx = jnp.zeros(self.n)
        for j in range(self.o - i - 1):
            dxj = (jnp.prod(jnp.cos(xx[:j] * jnp.pi / 2)) *
                   (-jnp.pi / 2) * jnp.sin(xx[j] * jnp.pi / 2) *
                   jnp.prod(jnp.cos(xx[j + 1:self.o - i - 1] * jnp.pi / 2)) *
                   (1 + ssx))
            if i > 0:
                dxj *= jnp.sin(xx[self.o - i - 1] * jnp.pi / 2)
            dx = dx.at[j].set(dxj)
        if i > 0:
            j = self.o - i - 1
            dxj = (jnp.prod(jnp.cos(xx[:self.o-i-1] * jnp.pi / 2)) *
                   (1 + ssx) * (jnp.pi / 2) *
                   jnp.cos(xx[self.o - i - 1] * jnp.pi / 2))
            dx = dx.at[j].set(dxj)
        # Evaluate derivative wrt ssx
        ds = jnp.prod(jnp.cos(xx[:self.o - i - 1] * jnp.pi / 2))
        if i > 0:
            ds *= jnp.sin(jnp.pi * xx[self.o - i - 1] / 2)
        return from_array(dx, self.des_type), from_array(ds, self.sim_type)


class dtlz3_grad(dtlz3_obj):
    """ Class defining the DTLZ3 gradient.

    Inherits from ``dtlz3_obj``, but overwrites the ``__call__`` method to
    evaluate the gradient wrt x and sx, respectively.

    """

    def __call__(self, x, sx):
        """ DTLZ3 gradient evaluation.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            dict, dict: The gradient of this objective for the input (x, sx)
            wrt x and sx, respectively.

        """

        # Extract x into xx and sx into ssx, if names are used
        xx = to_array(x, self.des_type)
        ssx = to_array(sx, self.sim_type)[0]
        # Evaluate derivative wrt xx
        i = self.obj_ind
        dx = jnp.zeros(self.n)
        for j in range(self.o - i - 1):
            dxj = (jnp.prod(jnp.cos(xx[:j] * jnp.pi / 2)) *
                   (-jnp.pi / 2) * jnp.sin(xx[j] * jnp.pi / 2) *
                   jnp.prod(jnp.cos(xx[j + 1:self.o - i - 1] * jnp.pi / 2)) *
                   (1 + ssx))
            if i > 0:
                dxj *= jnp.sin(xx[self.o - i - 1] * jnp.pi / 2)
            dx = dx.at[j].set(dxj)
        if i > 0:
            j = self.o - i - 1
            dxj = (jnp.prod(jnp.cos(xx[:self.o-i-1] * jnp.pi / 2)) *
                   (1 + ssx) * (jnp.pi / 2) *
                   jnp.cos(xx[self.o - i - 1] * jnp.pi / 2))
            dx = dx.at[j].set(dxj)
        # Evaluate derivative wrt ssx
        ds = jnp.prod(jnp.cos(xx[:self.o - i - 1] * jnp.pi / 2))
        if i > 0:
            ds *= jnp.sin(jnp.pi * xx[self.o - i - 1] / 2)
        return from_array(dx, self.des_type), from_array(ds, self.sim_type)


class dtlz4_grad(dtlz4_obj):
    """ Class defining the DTLZ4 gradient.

    Inherits from ``dtlz4_obj``, but overwrites the ``__call__`` method to
    evaluate the gradient wrt x and sx, respectively.

    """

    def __call__(self, x, sx):
        """ DTLZ4 gradient evaluation.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            dict, dict: The gradient of this objective for the input (x, sx)
            wrt x and sx, respectively.

        """

        # Extract x into xx and sx into ssx, if names are used
        xx = to_array(x, self.des_type)
        ssx = to_array(sx, self.sim_type)[0]
        # Evaluate derivative wrt xx
        i = self.obj_ind
        dx = jnp.zeros(self.n)
        for j in range(self.o - i - 1):
            dxj = (jnp.prod(jnp.cos(xx[:j] ** self.alpha * jnp.pi / 2))
                   * (-jnp.pi * self.alpha / 2)
                   * xx[j] ** (self.alpha - 1)
                   * jnp.sin(xx[j] ** self.alpha * jnp.pi / 2) *
                   jnp.prod(jnp.cos(xx[j+1:self.o-i-1] ** self.alpha *
                            jnp.pi / 2)) * (1 + ssx))
            if i > 0:
                dxj *= jnp.sin(xx[self.o - i - 1] ** self.alpha * jnp.pi / 2)
            dx = dx.at[j].set(dxj)
        if i > 0:
            j = self.o - i - 1
            dxj = (jnp.prod(jnp.cos(xx[0:self.o-i-1] ** self.alpha *
                            jnp.pi / 2)) * (1 + ssx) *
                   (jnp.pi * self.alpha / 2) *
                   xx[j] ** (self.alpha - 1) *
                   jnp.cos(xx[j] ** self.alpha * jnp.pi / 2))
            dx = dx.at[j].set(dxj)
        # Evaluate derivative wrt ssx
        ds = jnp.prod(jnp.cos(xx[:self.o - i - 1] ** self.alpha *
                              jnp.pi / 2))
        if i > 0:
            ds *= jnp.sin(xx[self.o - i - 1] ** self.alpha * jnp.pi / 2)
        return from_array(dx, self.des_type), from_array(ds, self.sim_type)

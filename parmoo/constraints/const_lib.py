""" This module contains a library of common constraint functions, matching
ParMOO's interface.

The common constraints are:
 * ``SingleSimBound`` -- bound on a single simulation output's value
 * ``SumOfSimSquaresBound`` -- bound on the SOS for several sim outputs
 * ``SumOfSimsBound`` -- bound on the (abs) sum of several sim outputs

And their corresponding gradient functions are:
 * ``SingleSimBoundGradient``
 * ``SumOfSimSquaresBoundGradient``
 * ``SumOfSimsBoundGradient``

"""

from jax import numpy as jnp
import numpy as np
from parmoo.structs import CompositeFunction


class SingleSimBound(CompositeFunction):
    """ Class for bounding a single simulation's output.

    If upper-bounding:

    ```
    def ConstraintFunction(x, sx):
        return sx[self.sim_ind] - upper_bound
    ```

    If lower-bounding:

    ```
    def ConstraintFunction(x, sx):
        return lower_bound - sx[self.sim_ind]
    ```

    Also supports derivative usage.

    Contains 2 methods:
     * ``__init__(des_type, sim_type, sim_ind, bound_type='min', bound=0)``
     * ``__call__(x, sim)``

    The ``__init__`` method inherits from the CompositeFunction ABC.

    The ``__call__`` returns the slack (negative when feasible).

    """

    def __init__(self, des_type, sim_type, sim_ind,
                 bound_type='upper', bound=0.0):
        """ Constructor for SingleSimBound class.

        Args:
            des_type (np.dtype): The numpy.dtype of the design variables.

            sim_type (np.dtype): The numpy.dtype of the simulation outputs.

            sim_ind (str or tuple): The name (or name, output index pair
                for simulations with more than one output field)
                designating the simulation output to minimize or maximize.

            bound_type (str): Either 'lower' to lower-bound or 'upper' to
                upper-bound. Defaults to 'upper'.

            bound (float): The lower/upper bound for this constraint.
                Defaults to 0.

        """

        super().__init__(des_type, sim_type)
        # Check additional inputs
        if isinstance(sim_ind, str):
            try:
                assert (sim_ind in np.dtype(self.sim_type).names)
            except BaseException:
                raise ValueError(f"{sim_ind[0]} not a name in given sim_type")
            self.sim_name = sim_ind
            self.goal = jnp.ones(1)
        elif isinstance(sim_ind, tuple):
            try:
                assert sim_ind[0] in np.dtype(self.sim_type).names
            except BaseException:
                raise ValueError(f"{sim_ind[0]} not a name in given sim_type")
            try:
                assert 0 <= sim_ind[1] < self.sim_type[sim_ind[0]].shape[0]
            except BaseException:
                raise ValueError(f"{sim_ind[1]} not an index of {sim_ind[0]}")
            self.sim_name = sim_ind[0]
            sim_size = self.sim_type[sim_ind[0]].shape[0]
            self.goal = jnp.eye(sim_size)[sim_ind[1]]
        else:
            raise TypeError("sim_ind must be a str or (str, int) tuple")
        # Check for optional input
        self.bound = bound
        if bound_type.strip().lower() not in ('lower', 'upper'):
            raise ValueError("bound type must be 'upper' or 'lower'")
        if bound_type.strip().lower() == 'lower':
            self.goal *= -1.0
            self.bound *= -1.0
        return

    def __call__(self, x, sx):
        """ Define constraint evaluation.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            float: The (negative when feasible) slack in this constraint
            for the input sx (x is unused).

        """

        # These lines look silly, but do not change them.
        # They are needed to gracefully cast all possible shapes to scalar.
        fx = 0.0
        fx += jnp.dot(sx[self.sim_name], self.goal)
        return fx - self.bound


class SumOfSimSquaresBound(CompositeFunction):
    """ Class for constraining the sum-of-squared simulation outputs.

    If upper bounding:

    ```
    ConstraintFunction(x, sx):
        return sum([sx[i]**2 for all i]) - upper_bound
    ```

    If lower bounding:

    ```
    def ConstraintFunction(x, sx):
        return lower_bound - sum([sx[i]**2 for all i])
    ```

    Also supports derivative usage.

    Contains 2 methods:
     * ``__init__(des_type, sim_type, sim_inds, bound_type='upper', bound=0)``
     * ``__call__(x, sx)``

    The ``__init__`` method inherits from the CompositeFunction ABC.

    The ``__call__`` evaluate the slack (negative values are feasible).

    """

    def __init__(self, des_type, sim_type, sim_inds,
                 bound_type='upper', bound=0.0):
        """ Constructor for SumOfSimSquaresBound class.

        Args:
            des_type (np.dtype): The numpy.dtype of the design variables.

            sim_type (np.dtype): The numpy.dtype of the simulation outputs.

            sim_inds (list): The list of names (or name, index pairs for sims
                with more than one output) of the simulation outputs to sum
                over.

            bound_type (str): Either 'lower' to lower-bound or 'upper' to
                upper-bound. Defaults to 'upper'.

            bound (float): The lower/upper bound for this constraint.
                Defaults to 0.

        """

        super().__init__(des_type, sim_type)
        # Check additional inputs
        if not isinstance(sim_inds, list):
            raise TypeError("sim_inds must be a list of ints, tuples, or " +
                            "strings")
        self.sim_names = []
        self.sim_inds = []
        for sim_ind in sim_inds:
            if isinstance(sim_ind, str):
                try:
                    assert (sim_ind in np.dtype(self.sim_type).names)
                except BaseException:
                    raise ValueError(f"{sim_ind[0]} not in given sim_type")
                if sim_ind in self.sim_names:
                    self.sim_inds[self.sim_inds.index(sim_ind)] += 1.0
                else:
                    self.sim_names.append(sim_ind)
                    self.sim_inds.append(jnp.ones(1))
            elif isinstance(sim_ind, tuple):
                try:
                    assert sim_ind[0] in np.dtype(self.sim_type).names
                except BaseException:
                    raise ValueError(f"{sim_ind[0]} not in given sim_type")
                try:
                    assert 0 <= sim_ind[1] < self.sim_type[sim_ind[0]].shape[0]
                except BaseException:
                    raise ValueError(f"{sim_ind[1]} not an index of "
                                     f"{sim_ind[0]}")
                sim_size = self.sim_type[sim_ind[0]].shape[0]
                if sim_ind[0] in self.sim_names:
                    self.sim_inds[self.sim_names.index(sim_ind[0])] += \
                            jnp.eye(sim_size)[sim_ind[1]]
                else:
                    self.sim_names.append(sim_ind[0])
                    self.sim_inds.append(jnp.eye(sim_size)[sim_ind[1]])
            else:
                raise TypeError("Each sim_ind must be a str or str,int tuple")
        # Check for optional input
        self.bound = bound
        if bound_type.lower() not in ('lower', 'upper'):
            raise ValueError("bound type must be 'upper' or 'lower'")
        if bound_type.lower() == 'lower':
            self.goal = -1.0
            self.bound *= -1.0
        else:
            self.goal = 1.0
        return

    def __call__(self, x, sx):
        """ Define constraint evaluation.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            float: The (negative when feasible) slack in this constraint
            for the input sx (x is unused).

        """

        fx = 0.0
        for sn, si in zip(self.sim_names, self.sim_inds):
            fx += jnp.dot(sx[sn] ** 2, si)
        return fx * self.goal - self.bound


class SumOfSimsBound(CompositeFunction):
    """ Class for bounding the (absolute) sum of simulation outputs.

    If upper bounding:

    ```
    def const_func(x, sx):
        return sum([sx[i] for all i]) - upper_bound
    ```

    If lower bounding:

    ```
    def const_func(x, sx):
        return lower_bound - sum([sx[i] for all i])
    ```

    If upper bounding absolute sum:

    ```
    def const_func(x, sx):
        return sum([abs(sx[i]) for all i]) - upper_bound
    ```

    If lower bounding absolute sum:

    ```
    def const_func(x, sx):
        return lower_bound - sum([abs(sx[i]) for all i])
    ```

    Also supports derivative usage.

    Contains 2 methods:
     * ``__init__(des_type, sim_type, sim_inds,
                  bound_type='upper', bound=0, absolute=False)``
     * ``__call__(x, sim)``

    The ``__init__`` method inherits from the CompositeFunction ABC.

    The ``__call__`` evaluate the slack (negative values are feasible).

    """

    def __init__(self, des_type, sim_type, sim_inds,
                 bound_type='upper', bound=0.0, absolute=False):
        """ Constructor for SumOfSimsBound class.

        Args:
            des_type (np.dtype): The numpy.dtype of the design variables.

            sim_type (np.dtype): The numpy.dtype of the simulation outputs.

            sim_inds (list): The list of names (or name, index pairs for sims
                with more than one output) of the simulation outputs to sum
                over.

            bound_type (str): Either 'lower' to lower-bound or 'upper' to
                upper-bound. Defaults to 'upper'.

            bound (float): The lower/upper bound for this constraint.
                Defaults to 0.

            absolute (bool): True to bound absolute sum, False to
                bound raw sum. Defaults to False.

        """

        super().__init__(des_type, sim_type)
        # Check additional inputs
        if not isinstance(sim_inds, list):
            raise TypeError("sim_inds must be a list of ints, tuples, or " +
                            "strings")
        self.sim_names = []
        self.sim_inds = []
        for sim_ind in sim_inds:
            if isinstance(sim_ind, str):
                try:
                    assert (sim_ind in np.dtype(self.sim_type).names)
                except BaseException:
                    raise ValueError(f"{sim_ind[0]} not in given sim_type")
                if sim_ind in self.sim_names:
                    self.sim_inds[self.sim_inds.index(sim_ind)] += 1.0
                else:
                    self.sim_names.append(sim_ind)
                    self.sim_inds.append(jnp.ones(1))
            elif isinstance(sim_ind, tuple):
                try:
                    assert sim_ind[0] in np.dtype(self.sim_type).names
                except BaseException:
                    raise ValueError(f"{sim_ind[0]} not in given sim_type")
                try:
                    assert 0 <= sim_ind[1] < self.sim_type[sim_ind[0]].shape[0]
                except BaseException:
                    raise ValueError(f"{sim_ind[1]} not an index of "
                                     f"{sim_ind[0]}")
                sim_size = self.sim_type[sim_ind[0]].shape[0]
                if sim_ind[0] in self.sim_names:
                    self.sim_inds[self.sim_names.index(sim_ind[0])] += \
                            jnp.eye(sim_size)[sim_ind[1]]
                else:
                    self.sim_names.append(sim_ind[0])
                    self.sim_inds.append(jnp.eye(sim_size)[sim_ind[1]])
            else:
                raise TypeError("Each sim_ind must be a str or str,int tuple")
        # Check for optional input
        self.bound = bound
        if bound_type.lower() not in ('lower', 'upper'):
            raise ValueError("bound type must be 'upper' or 'lower'")
        if bound_type.lower() == 'lower':
            self.goal = -1.0
            self.bound *= -1.0
        else:
            self.goal = 1.0

        def id_func(x): return np.ones(x.size)

        def abs_func(x): return jnp.sign(x)

        if not isinstance(absolute, bool):
            raise TypeError("absolute must be a bool type, not " +
                            str(type(absolute)))
        if absolute:
            self.absolute = abs_func
        else:
            self.absolute = id_func
        return

    def __call__(self, x, sx):
        """ Define constraint evaluation.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            float: The (negative when feasible) slack in this constraint
            for the input sx (x is unused).

        """

        fx = 0.0
        for sn, si in zip(self.sim_names, self.sim_inds):
            fx += jnp.sum(self.absolute(sx[sn]) * sx[sn] * si)
        return fx * self.goal - self.bound


class SingleSimBoundGradient(SingleSimBound):
    """ Gradient class for SingleSimBound.

    Inherits from the ``SingleSimBound`` class, but overwrites the
    ``__call__`` method to return the gradients wrt x and sx, respectively.

    """

    def __call__(self, x, sx):
        """ Define gradient evaluation.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            dict, dict: The gradient output of the simulation at self.sim_ind
            given the input (x, sx) wrt x and sx, respectively.

        """

        dx, ds = {}, {}
        for name in self.des_type.names:
            dx[name] = jnp.zeros(1)
        for name in self.sim_type.names:
            size = max(sum(self.sim_type[name].shape), 1)
            ds[name] = jnp.zeros(size)
        ds[self.sim_name] = self.goal
        return dx, ds


class SumOfSimSquaresBoundGradient(SumOfSimSquaresBound):
    """ Gradient class for SumOfSimSquaresBound.

    Inherits from the ``SumOfSimSquaresBound`` class, but overwrites the
    ``__call__`` method to return the gradients wrt x and sx, respectively.

    """

    def __call__(self, x, sx):
        """ Define gradient evaluation.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            dict, dict: The gradient of the sum of squared simulation outputs
            given the input (x, sx) wrt x and sx, respectively.

        """

        dx, ds = {}, {}
        for name in self.des_type.names:
            dx[name] = jnp.zeros(1)
        for name in self.sim_type.names:
            size = max(sum(self.sim_type[name].shape), 1)
            ds[name] = jnp.zeros(size)
        for sn, si in zip(self.sim_names, self.sim_inds):
            ds[sn] = sx[sn] * si * 2.0 * self.goal
        return dx, ds


class SumOfSimsBoundGradient(SumOfSimsBound):
    """ Gradient class for SumOfSimsBound.

    Inherits from the ``SumOfSimsBound`` class, but overwrites the
    ``__call__`` method to return the gradients wrt x and sx, respectively.

    """

    def __call__(self, x, sx):
        """ Define gradient evaluation.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            dict, dict: The gradient of the sum of (absolute) simulation
            outputs given the input (x, sx) wrt x and sx, respectively.

        """

        dx, ds = {}, {}
        for name in self.des_type.names:
            dx[name] = jnp.zeros(1)
        for name in self.sim_type.names:
            size = max(sum(self.sim_type[name].shape), 1)
            ds[name] = jnp.zeros(size)
        for sn, si in zip(self.sim_names, self.sim_inds):
            ds[sn] = self.absolute(sx[sn]) * si * self.goal
        return dx, ds

""" This module contains a library of common objective functions, matching
ParMOO's interface.

The common objectives are:
 * ``SingleSimObjective`` -- min or max a single simulation output
 * ``SumOfSimSquaresObjective`` -- min or max several squared sim outputs
 * ``SumOfSimsObjective`` -- min or max the (absolute) sum of sim outputs

And the corresponding gradients are defined in:
 * ``SingleSimGradient``
 * ``SumOfSimSquaresGradient``
 * ``SumOfSimsGradient``

"""

from jax import numpy as jnp
import numpy as np
from parmoo.structs import CompositeFunction


class SingleSimObjective(CompositeFunction):
    """ Class for min/maxing a single simulation's output.

    If minimizing:

    ``def ObjectiveFunction(x, sx): return sx[self.sim_ind]``

    If maximizing:

    ``def ObjectiveFunction(x, sx): return -sx[self.sim_ind]``

    Also supports derivative usage.

    Contains 2 methods:
     * ``__init__(des_type, sim_type, sim_ind, goal='min')``
     * ``__call__(x, sim)``

    The ``__init__`` method inherits from the CompositeFunction ABC.

    The ``__call__`` returns sim[self.sim_ind].

    """

    def __init__(self, des_type, sim_type, sim_ind, goal='min'):
        """ Constructor for SingleSimObjective class.

        Args:
            des_type (np.dtype): The numpy.dtype of the design variables.

            sim_type (np.dtype): The numpy.dtype of the simulation outputs.

            sim_ind (str or tuple): The name (or name, output index pair
                for simulations with more than one output field)
                designating the simulation output to minimize or maximize.

            goal (str): Either 'min' to minimize or 'max' to maximize.
                Defaults to 'min'.

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
        if goal.lower() not in ('min', 'max'):
            raise ValueError("goal must be either 'min' or 'max'")
        if goal.lower() == 'max':
            self.goal *= -1.0
        return

    def __call__(self, x, sx):
        """ Define objective evaluation.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            float: The output of the simulation at self.sim_ind given the
            input x.

        """

        # These lines look silly, but do not change them.
        # They are needed to gracefully casts all possible shapes to scalar.
        fx = 0.0
        fx += jnp.dot(sx[self.sim_name], self.goal)
        return fx


class SumOfSimSquaresObjective(CompositeFunction):
    """ Class for min/maxing the sum-of-squared simulation outputs.

    If minimizing, equivalent to:

    ```
    def ObjectiveFunction(x, sx):
        return sum([sx[i]**2 for i in sim_inds])
    ```

    If maximizing, equivalent to:

    ```
    def ObjectiveFunction(x, sx):
        return -sum([sx[i]**2 for i in sim_inds])
    ```

    Also supports derivative usage.

    Contains 2 methods:
     * ``__init__(des_type, sim_type, sim_inds, goal='min')``
     * ``__call__(x, sx)``

    The ``__init__`` method inherits from the CompositeFunction ABC.

    The ``__call__`` evaluates the sum-of-squared simulation outputs.

    """

    def __init__(self, des_type, sim_type, sim_inds, goal='min'):
        """ Constructor for SumOfSimSquaresObjective class.

        Args:
            des_type (np.dtype): The numpy.dtype of the design variables.

            sim_type (np.dtype): The numpy.dtype of the simulation outputs.

            sim_inds (list): The list of names (or name, index pairs for sims
                with more than one output) of the simulation outputs to sum
                over.

            goal (str): Either 'min' to minimize SOS or 'max' to maximize SOS.
                Defaults to 'min'.

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
        if goal.lower() not in ('min', 'max'):
            raise ValueError("goal must be either 'min' or 'max'")
        if goal.lower() == 'max':
            self.goal = -1.0
        else:
            self.goal = 1.0
        return

    def __call__(self, x, sx):
        """ Define objective evaluation.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            float: The sum of squared outputs of the simulation at indices in
            self.sim_ind given the input x.

        """

        fx = 0.0
        for sn, si in zip(self.sim_names, self.sim_inds):
            fx += jnp.dot(sx[sn] ** 2, si)
        return fx * self.goal


class SumOfSimsObjective(CompositeFunction):
    """ Class for min/maxing the sum of simulation outputs.

    If minimizing:

    ``def ObjectiveFunction(x, sx): return sum([sx[i] for i in sim_inds])``

    If maximizing:

    ``def ObjectiveFunction(x, sx): return -sum([sx[i] for i in sim_inds])``

    If minimizing absolute sum:

    ```
    def ObjectiveFunction(x, sx): return sum([abs(sx[i]) for i in sim_inds])
    ```

    If maximizing absolute sum:

    ```
    def ObjectiveFunction(x, sx): return -sum([abs(sx[i]) for i in sim_inds])
    ```

    Also supports derivative usage.

    Contains 2 methods:
     * ``__init__(des_type, sim_type, sim_inds, goal='min', absolute=False)``
     * ``__call__(x, sx)``

    The ``__init__`` method inherits from the CompositeFunction ABC.

    The ``__call__`` evaluate the (absolute) sum outputs.

    """

    def __init__(self, des_type, sim_type, sim_inds, goal='min',
                 absolute=False):
        """ Constructor for SumOfSimsObjective class.

        Args:
            des_type (np.dtype): The numpy.dtype of the design variables.

            sim_type (np.dtype): The numpy.dtype of the simulation outputs.

            sim_inds (list): The list of indices or names of the
                simulation outputs to sum over.

            goal (str): Either 'min' to minimize sum or 'max' to maximize sum.
                Defaults to 'min'.

            absolute (bool): True to min/max absolute sum, False to
                min/max raw sum. Defaults to False.

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
        if goal.lower() not in ('min', 'max'):
            raise ValueError("goal must be either 'min' or 'max'")
        if goal.lower() == 'max':
            self.goal = -1.0
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
        """ Define objective evaluation.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            float: The sum of the simulation outputs at indices in
            self.sim_ind given the input x.

        """

        fx = 0.0
        for sn, si in zip(self.sim_names, self.sim_inds):
            fx += jnp.sum(self.absolute(sx[sn]) * sx[sn] * si)
        return fx * self.goal


class SingleSimGradient(SingleSimObjective):
    """ Gradient class for SingleSimObjective.

    Inherits from the ``SingleSimObjective`` class, but overwrites the
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


class SumOfSimSquaresGradient(SumOfSimSquaresObjective):
    """ Gradient class for SumOfSimSquaresObjective.

    Inherits from the ``SumOfSimSquaresObjective`` class, but overwrites the
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


class SumOfSimsGradient(SumOfSimsObjective):
    """ Gradient class for SumOfSimsObjective.

    Inherits from the ``SumOfSimsObjective`` class, but overwrites the
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

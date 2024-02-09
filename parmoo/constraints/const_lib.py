""" This module contains a library of common constraint functions, matching
ParMOO's interface.

The common constraints are:
 * ``SingleSimBound`` -- min or max bound on a single simulation output
 * ``SumOfSimSquaresBound`` -- min or max bound on the SOS for several sim outputs
 * ``sum_sim_out`` -- min or max bound on the (abs) sum of several sim outputs

"""

from parmoo.structs import CompositeFunction
import numpy as np


class SingleSimBound(CompositeFunction):
    """ Class for bounding a single simulation's output.

    Upper or lower bound a single simulation output.

    If upper-bounding:

    ```
    def ConstraintFunction(x, sx, der=0):
        return sx[self.sim_ind] - upper_bound
    ```

    If lower-bounding:

    ```
    def ConstraintFunction(x, sx, der=0):
        return lower_bound - sx[self.sim_ind]
    ```

    Also supports derivative usage.

    Contains 2 methods:
     * ``__init__(des, sim, sim_ind, type='min', bound=0.0)``
     * ``__call__(x, sim, der=0)``

    The ``__init__`` method inherits from the CompositeFunction ABC.

    The ``__call__`` returns the slack (negative when feasible).

    """

    def __init__(self, des, sim, sim_ind, type='upper', bound=0.0):
        """ Constructor for SingleSimBound class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be unnamed.

            sim (np.dtype or int): Either the numpy.dtype of the
                simulation outputs or the number of simulation outputs,
                assumed to all be unnamed.

            sim_ind (int, str, or tuple): The index or name of the simulation
                output to minimize or maximize. Use an integer index
                when des & sim contain unnamed types. Use a str name when
                des & sim contain named types. Use a tuple when
                sim[sim_ind] has multiple outputs, where the first
                entry is the name of the simulation, and the
                second entry is the index of that simulation's output
                to minimize/maximize.

            type (str): Either 'lower' to lower-bound or 'upper' to
                upper-bound. Defaults to 'upper'.

            bound (float): The lower/upper bound for this constraint.
                Defaults to 0.

        """

        super().__init__(des, sim)
        # Check additional inputs
        if isinstance(sim_ind, str):
            try:
                assert(sim_ind in np.dtype(self.sim_type).names)
            except BaseException:
                raise ValueError(sim_ind + " is not a legal name in " +
                                 str(np.dtype(sim)))
        elif isinstance(sim_ind, tuple):
            try:
                assert(sim_ind[0] in np.dtype(self.sim_type).names)
            except BaseException:
                raise ValueError(str(sim_ind[0]) + " is not a legal name in "
                                 + str(np.dtype(sim)))
        elif isinstance(sim_ind, int):
            if self.sim_type.names is not None:
                raise TypeError("Type mismatch: " + str(sim_ind) + " and " +
                                str(np.dtype(self.sim_type)))
            elif sim_ind < 0 or sim_ind > self.m:
                raise ValueError(str(sim_ind) + " is not a valid index" +
                                 " in the range: [0, " + str(self.m - 1)
                                 + "]")
        else:
            raise TypeError("sim_ind must have the int, str, or tuple type")
        self.sim_ind = sim_ind
        # Check for optional input
        if type.lower() not in ('lower', 'upper'):
            raise ValueError("bound type must be 'upper' or 'lower', not '" +
                             str(type) + "'")
        if type.lower() == 'upper':
            self.type = 1.0
        else:
            self.type = -1.0
        self.bound = bound
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
            float or numpy.array: The (negative when feasible) slack in
            this constraint for the input x (der=0), the gradient with
            respect to x (der=1), or the gradient with respect to sim (der=2).

        """

        # Evaluate derivative wrt x
        if der == 1:
            return np.zeros(1, dtype=self.des_type)[0]
        # Evaluate derivative wrt sim
        elif der == 2:
            ds = np.zeros(1, dtype=self.sim_type)[0]
            if isinstance(self.sim_ind, tuple):
                ds[self.sim_ind[0]][self.sim_ind[1]] = self.type
            else:
                ds[self.sim_ind] = self.type
            return ds
        # Evaluate g(x, sim)
        else:
            if isinstance(self.sim_ind, tuple):
                return (sim[self.sim_ind[0]][self.sim_ind[1]] - self.bound) \
                       * self.type
            else:
                return (sim[self.sim_ind] - self.bound) * self.type


class SumOfSimSquaresBound(CompositeFunction):
    """ Class for constraining the sum-of-squared simulation outputs.

    Upper or lower bound the sum-of-squared simulation outputs.

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
     * ``__init__(des, sim, sim_inds, type='upper', bound=0.0)``
     * ``__call__(x, sx, der=0)``

    The ``__init__`` method inherits from the CompositeFunction ABC.

    The ``__call__`` evaluate the slack (negative values are feasible).

    """

    def __init__(self, des, sim, sim_inds, type='upper', bound=0.0):
        """ Constructor for SumOfSimSquaresBound class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be unnamed.

            sim (np.dtype or int): Either the numpy.dtype of the
                simulation outputs or the number of simulation outputs,
                assumed to all be unnamed.

            sim_inds (list): The list of indices or names of the
                simulation outputs to sum over.

            type (str): Either 'lower' to lower-bound or 'upper' to
                upper-bound. Defaults to 'upper'.

            bound (float): The lower/upper bound for this constraint.
                Defaults to 0.

        """

        super().__init__(des, sim)
        # Check additional inputs
        if not isinstance(sim_inds, list):
            raise TypeError("sim_inds must be a list of ints, tuples, or " +
                            "strings")
        if all([isinstance(si, str) or isinstance(si, tuple)
                for si in sim_inds]):
            for si in sim_inds:
                if isinstance(si, str):
                    try:
                        assert(si in np.dtype(self.sim_type).names)
                    except BaseException:
                        raise ValueError(si + " is not a legal name in " +
                                         str(np.dtype(sim)))
                else:
                    try:
                        assert(si[0] in np.dtype(self.sim_type).names)
                    except BaseException:
                        raise ValueError(str(si[0]) +
                                         " is not a legal name in " +
                                         str(np.dtype(sim)))
        elif all([isinstance(si, tuple) for si in sim_inds]):
            try:
                for si in sim_inds:
                    assert(si[0] in self.sim_type.names)
            except BaseException:
                raise ValueError(si[0] + " is not a legal name in " +
                                 str(np.dtype(sim)))
        elif all([isinstance(si, int) for si in sim_inds]):
            if self.sim_type.names is not None:
                raise TypeError("Type mismatch: int and " +
                                str(np.dtype(self.sim_type)))
            elif any([si < 0 or si > self.m for si in sim_inds]):
                raise ValueError(str(sim_inds) + " contains invalid indices")
        else:
            raise TypeError("sim_inds must be a list of ints, tuples, or " +
                            "strings")
        self.sim_inds = sim_inds
        # Check for optional input
        if type.lower() not in ('lower', 'upper'):
            raise ValueError("bound type must be 'upper' or 'lower', not '" +
                             str(type) + "'")
        if type.lower() == 'upper':
            self.type = 1.0
        else:
            self.type = -1.0
        if not isinstance(bound, float) and not isinstance(bound, int):
            raise TypeError("The upper/lower bound must be a numeric type")
        self.bound = bound
        return

    def __call__(self, x, sim, der=0):
        """ Define constraint evaluation.

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
            float or numpy.array: The (negative when feasible) slack in
            this constraint for the input x (der=0), the gradient with
            respect to x (der=1), or the gradient with respect to sim (der=2).

        """

        # Evaluate derivative wrt x
        if der == 1:
            return np.zeros(1, dtype=self.des_type)[0]
        # Evaluate derivative wrt sim
        elif der == 2:
            ds = np.zeros(1, dtype=self.sim_type)[0]
            for si in self.sim_inds:
                if isinstance(si, tuple):
                    ds[si[0]][si[1]] = sim[si[0]][si[1]] * 2.0 * self.type
                else:
                    ds[si] = sim[si] * 2.0 * self.type
            return ds
        # Evaluate f(x, sim)
        else:
            fx = 0.0
            for si in self.sim_inds:
                if isinstance(si, tuple):
                    fx += sim[si[0]][si[1]] ** 2
                else:
                    try:
                        fx += sum(sim[si] ** 2)
                    except TypeError:
                        fx += sim[si] ** 2
            return (fx - self.bound) * self.type


class sum_sim_bound(CompositeFunction):
    """ Class for bounding the sum of simulation outputs.

    Upper or lower bound the (absolute) sum of simulation output.

    If upper bounding:

    ``def const_func(x, sx): return sum([sx[i] for all i]) - upper_bound``

    If lower bounding:

    ``def const_func(x, sx): return lower_bound - sum([sx[i] for all i])``

    If upper bounding absolute sum:

    ``def const_func(x, sx): return sum([abs(sx[i]) forall i]) - upper_bound``

    If lower bounding absolute sum:

    ``def const_func(x, sx): return lower_bound - sum([abs(sx[i]) forall i])``

    Also supports derivative usage.

    Contains 2 methods:
     * ``__init__(des, sim, sim_inds, type='upper', bound=0, absolute=False)``
     * ``__call__(x, sim, der=0)``

    The ``__init__`` method inherits from the CompositeFunction ABC.

    The ``__call__`` evaluate the slack (negative values are feasible).

    """

    def __init__(self, des, sim, sim_inds,
                 type='upper', bound=0.0, absolute=False):
        """ Constructor for sum_sim_bound class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be unnamed.

            sim (np.dtype or int): Either the numpy.dtype of the
                simulation outputs or the number of simulation outputs,
                assumed to all be unnamed.

            sim_inds (list): The list of indices or names of the
                simulation outputs to sum over.

            type (str): Either 'lower' to lower-bound or 'upper' to
                upper-bound. Defaults to 'upper'.

            bound (float): The lower/upper bound for this constraint.
                Defaults to 0.

            absolute (bool): True to bound absolute sum, False to
                bound raw sum. Defaults to False.

        """

        super().__init__(des, sim)
        # Check additional inputs
        if not isinstance(sim_inds, list):
            raise TypeError("sim_inds must be a list of ints, tuples, or " +
                            "strings")
        if all([isinstance(si, str) or isinstance(si, tuple)
                for si in sim_inds]):
            for si in sim_inds:
                if isinstance(si, str):
                    try:
                        assert(si in np.dtype(self.sim_type).names)
                    except BaseException:
                        raise ValueError(si + " is not a legal name in " +
                                         str(np.dtype(sim)))
                else:
                    try:
                        assert(si[0] in np.dtype(self.sim_type).names)
                    except BaseException:
                        raise ValueError(str(si[0]) +
                                         " is not a legal name in " +
                                         str(np.dtype(sim)))
        elif all([isinstance(si, tuple) for si in sim_inds]):
            try:
                for si in sim_inds:
                    assert(si[0] in self.sim_type.names)
            except BaseException:
                raise ValueError(si[0] + " is not a legal name in " +
                                 str(np.dtype(sim)))
        elif all([isinstance(si, int) for si in sim_inds]):
            if self.sim_type.names is not None:
                raise TypeError("Type mismatch: int and " +
                                str(np.dtype(self.sim_type)))
            elif any([si < 0 or si > self.m for si in sim_inds]):
                raise ValueError(str(sim_inds) + " contains invalid indices")
        else:
            raise TypeError("sim_inds must be a list of ints, tuples, or " +
                            "strings")
        self.sim_inds = sim_inds
        # Check for optional input
        if type.lower() not in ('lower', 'upper'):
            raise ValueError("bound type must be 'upper' or 'lower', not '" +
                             str(type) + "'")
        if type.lower() == 'upper':
            self.type = 1.0
        else:
            self.type = -1.0
        if not isinstance(bound, float) and not isinstance(bound, int):
            raise TypeError("The upper/lower bound must be a numeric type")
        self.bound = bound
        if not isinstance(absolute, bool):
            raise TypeError("absolute must be a bool type, not " +
                            str(type(absolute)))
        self.absolute = absolute
        return

    def __call__(self, x, sim, der=0):
        """ Define constraint evaluation.

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
            float or numpy.array: The (negative when feasible) slack in
            this constraint for the input x (der=0), the gradient with
            respect to x (der=1), or the gradient with respect to sim (der=2).

        """

        # Evaluate derivative wrt x
        if der == 1:
            return np.zeros(1, dtype=self.des_type)[0]
        # Evaluate derivative wrt sim
        elif der == 2:
            ds = np.zeros(1, dtype=self.sim_type)[0]
            for si in self.sim_inds:
                if isinstance(si, tuple):
                    if self.absolute and sim[si[0]][si[1]] < 0.0:
                        ds[si[0]][si[1]] = -1.0 * self.type
                    else:
                        ds[si[0]][si[1]] = 1.0 * self.type
                else:
                    if self.absolute and sim[si] < 0.0:
                        ds[si] = -1.0 * self.type
                    else:
                        ds[si] = 1.0 * self.type
            return ds
        # Evaluate f(x, sim)
        else:
            fx = 0.0
            for si in self.sim_inds:
                if self.absolute:
                    if isinstance(si, tuple):
                        fx += abs(sim[si[0]][si[1]])
                    else:
                        try:
                            fx += sum(abs(sim[si]))
                        except TypeError:
                            fx += abs(sim[si])
                else:
                    if isinstance(si, tuple):
                        fx += sim[si[0]][si[1]]
                    else:
                        try:
                            fx += sum(sim[si])
                        except TypeError:
                            fx += sim[si]
            return (fx - self.bound) * self.type

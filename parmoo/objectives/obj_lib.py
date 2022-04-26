""" This module contains a library of common objective functions, matching
ParMOO's interface.

The common objectives are:
 * ``single_sim_out`` -- min or max a single simulation output
 * ``sos_sim_out`` -- min or max the sum-of-squares for several sim outputs
 * ``sum_sim_out`` -- min or max the (absolute) sum of several sim outputs

"""

from parmoo.objectives import obj_func
import numpy as np


class single_sim_out(obj_func):
    """ Class for optimizing a single simulation's output.

    Minimize or maximize a single simulation output. This simulation's
    value will be used as an objective.

    If minimizing:

    ``def obj_func(x, sx, der=0): return sx[self.sim_ind]``

    If maximizing:

    ``def obj_func(x, sx, der=0): return -sx[self.sim_ind]``

    Also supports derivative usage.

    Contains 2 methods:
     * ``__init__(des, sim, sim_ind, goal='min')``
     * ``__call__(x, sim, der=0)``

    The ``__init__`` method inherits from the obj_func ABC.

    The ``__call__`` returns sim[self.sim_ind].

    """

    def __init__(self, des, sim, sim_ind, goal='min'):
        """ Constructor for single_sim_out class.

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

            goal (str): Either 'min' to minimize or 'max' to maximize.
                Defaults to 'min'.

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
                raise ValueError(str(sim_ind[0]) + " is not a legal name in " +
                                 str(np.dtype(sim)))
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
        if goal.lower() not in ('min', 'max'):
            raise ValueError("goal must be 'min' or 'max', not '" +
                             str(goal) + "'")
        if goal.lower() == 'min':
            self.goal = 1.0
        else:
            self.goal = -1.0
        return

    def __call__(self, x, sim, der=0):
        """ Define objective evaluation.

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
            float or numpy.array: The output of this objective for the input
            x (der=0), the gradient with respect to x (der=1), or the
            gradient with respect to sim (der=2).

        """

        # Evaluate derivative wrt x
        if der == 1:
            return np.zeros(1, dtype=self.des_type)[0]
        # Evaluate derivative wrt sim
        elif der == 2:
            ds = np.zeros(1, dtype=self.sim_type)[0]
            if isinstance(self.sim_ind, tuple):
                ds[self.sim_ind[0]][self.sim_ind[1]] = self.goal
            else:
                ds[self.sim_ind] = self.goal
            return ds
        # Evaluate f(x, sim)
        else:
            if isinstance(self.sim_ind, tuple):
                return sim[self.sim_ind[0]][self.sim_ind[1]] * self.goal
            else:
                return sim[self.sim_ind] * self.goal


class sos_sim_out(obj_func):
    """ Class for optimizing the sum-of-squared simulation outputs.

    Minimize or maximize the sum-of-squared simulation outputs. This
    sum-of-squares (SOS) will be used as an objective.

    If minimizing:

    ``def obj_func(x, sx, der=0): return sum([sx[i]**2 for i in sim_inds])``

    If maximizing:

    ``def obj_func(x, sx, der=0): return -sum([sx[i]**2 for i in sim_inds])``

    Also supports derivative usage.

    Contains 2 methods:
     * ``__init__(des, sim, sim_inds, goal='min')``
     * ``__call__(x, sim, der=0)``

    The ``__init__`` method inherits from the obj_func ABC.

    The ``__call__`` evaluate the sum-of-square outputs.

    """

    def __init__(self, des, sim, sim_inds, goal='min'):
        """ Constructor for sos_sim_out class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be unnamed.

            sim (np.dtype or int): Either the numpy.dtype of the
                simulation outputs or the number of simulation outputs,
                assumed to all be unnamed.

            sim_inds (list): The list of indices or names of the
                simulation outputs to sum over.

            goal (str): Either 'min' to minimize SOS or 'max' to maximize SOS.
                Defaults to 'min'.

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
        if goal.lower() not in ('min', 'max'):
            raise ValueError("goal must be 'min' or 'max', not '" +
                             str(goal) + "'")
        if goal.lower() == 'min':
            self.goal = 1.0
        else:
            self.goal = -1.0
        return

    def __call__(self, x, sim, der=0):
        """ Define objective evaluation.

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
            float or numpy.array: The output of this objective for the input
            x (der=0), the gradient with respect to x (der=1), or the
            gradient with respect to sim (der=2).

        """

        # Evaluate derivative wrt x
        if der == 1:
            return np.zeros(1, dtype=self.des_type)[0]
        # Evaluate derivative wrt sim
        elif der == 2:
            ds = np.zeros(1, dtype=self.sim_type)[0]
            for si in self.sim_inds:
                if isinstance(si, tuple):
                    ds[si[0]][si[1]] = sim[si[0]][si[1]] * 2.0 * self.goal
                else:
                    ds[si] = sim[si] * 2.0 * self.goal
            return ds
        # Evaluate f(x, sim)
        else:
            fx = 0.0
            for si in self.sim_inds:
                if isinstance(si, tuple):
                    fx += sim[si[0]][si[1]] ** 2.0
                else:
                    fx += sim[si] ** 2.0
            return fx * self.goal


class sum_sim_out(obj_func):
    """ Class for optimizing the sum of simulation outputs.

    Minimize or maximize the (absolute) sum of simulation output.
    This sum will be used as an objective.

    If minimizing:

    ``def obj_func(x, sx, der=0): return sum([sx[i] for i in sim_inds])``

    If maximizing:

    ``def obj_func(x, sx, der=0): return -sum([sx[i] for i in sim_inds])``

    If minimizing absolute sum:

    ``def obj_func(x, sx, der=0): return sum([abs(sx[i]) for i in sim_inds])``

    If maximizing absolute sum:

    ``def obj_func(x, sx, der=0): return -sum([abs(sx[i]) for i in sim_inds])``

    Also supports derivative usage.

    Contains 2 methods:
     * ``__init__(des, sim, sim_inds, goal='min', absolute=False)``
     * ``__call__(x, sim, der=0)``

    The ``__init__`` method inherits from the obj_func ABC.

    The ``__call__`` evaluate the (absolute) sum outputs.

    """

    def __init__(self, des, sim, sim_inds, goal='min', absolute=False):
        """ Constructor for sum_sim_out class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be unnamed.

            sim (np.dtype or int): Either the numpy.dtype of the
                simulation outputs or the number of simulation outputs,
                assumed to all be unnamed.

            sim_inds (list): The list of indices or names of the
                simulation outputs to sum over.

            goal (str): Either 'min' to minimize sum or 'max' to maximize sum.
                Defaults to 'min'.

            absolute (bool): True to min/max absolute sum, False to
                min/max raw sum. Defaults to False.

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
        # Check for optional inputs
        if goal.lower() not in ('min', 'max'):
            raise ValueError("goal must be 'min' or 'max', not '" +
                             str(goal) + "'")
        if goal.lower() == 'min':
            self.goal = 1.0
        else:
            self.goal = -1.0
        if not isinstance(absolute, bool):
            raise TypeError("absolute must be a bool type, not " +
                            str(type(absolute)))
        self.absolute = absolute
        return

    def __call__(self, x, sim, der=0):
        """ Define objective evaluation.

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
            float or numpy.array: The output of this objective for the input
            x (der=0), the gradient with respect to x (der=1), or the
            gradient with respect to sim (der=2).

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
                        ds[si[0]][si[1]] = -1.0 * self.goal
                    else:
                        ds[si[0]][si[1]] = 1.0 * self.goal
                else:
                    if self.absolute and sim[si] < 0.0:
                        ds[si] = -1.0 * self.goal
                    else:
                        ds[si] = 1.0 * self.goal
            return ds
        # Evaluate f(x, sim)
        else:
            fx = 0.0
            for si in self.sim_inds:
                if self.absolute:
                    if isinstance(si, tuple):
                        fx += abs(sim[si[0]][si[1]])
                    else:
                        fx += abs(sim[si])
                else:
                    if isinstance(si, tuple):
                        fx += sim[si[0]][si[1]]
                    else:
                        fx += sim[si]
            return fx * self.goal

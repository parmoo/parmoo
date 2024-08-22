
""" Abstract base classes (ABCs) for ParMOO project.

This module contains several ABCs that can be used to extend and customize
ParMOO's solver components and/or supported problem types.

The classes include:
 * AcquisitionFunction
 * CompositeFunction
 * Embedder
 * GlobalSearch
 * SurrogateFunction
 * SurrogateOptimizer

"""

from abc import ABC, abstractmethod
import inspect
import numpy as np
from scipy.stats import tstd


class AcquisitionFunction(ABC):
    """ ABC describing acquisition functions.

    This class contains the following methods:
     * ``setTarget(data, penalty_func)``
     * ``scalarize(f_vals, x_vals, s_vals_mean, s_vals_sd)``
     * ``useSD()``
     * ``save(filename)``
     * ``load(filename)``

    """

    @abstractmethod
    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the AcquisitionFunction class.

        Args:
            o (int): The number of objectives.

            lb (ndarray): A 1D array of lower bounds for the design space.

            ub (ndarray): A 1D array of upper bounds for the design space.

            hyperparams (dict): A dictionary of hyperparameters that are
                passed to the acquisition function.

        Returns:
            AcquisitionFunction: A new AcquisitionFunction object.

        """

    @abstractmethod
    def setTarget(self, data, penalty_func):
        """ Set a new target value or region for the AcquisitionFunction.

        Args:
            data (dict): A dictionary specifying the current function
                evaluation database. It contains two mandatory fields:
                 * 'x_vals' (ndarray): A 2D array of design points.
                 * 'f_vals' (ndarray): A 2D array of corresponding objectives.

            penalty_func (function): A function of one (x) or two (x, sx)
                inputs that evaluates all (penalized) objective scores.

        Returns:
            ndarray: A 1D array containing a feasible starting point
            for the scalarized problem.

        """

    def useSD(self):
        """ Query whether this method uses uncertainties.

        When False, allows users to shortcut expensive uncertainty
        computations.

        Default implementation returns True, requiring full uncertainty
        computation for applicable models.

        """

        return True

    @abstractmethod
    def scalarize(self, f_vals, x_vals, s_vals_mean, s_vals_sd):
        """ Scalarize a vector-valued function using the AcquisitionFunction.

        Note: For best performance, make sure that jax can jit this method.

        Additionally, for compatibility with gradient-based solvers,
        this method must be implemented in jax and be differentiable
        via the jax.jacrev() tool.

        Args:
            f_vals (ndarray): A 1D array specifying a vector of function
                values to be scalarized.

            x_vals (ndarray): A 1D array specifying a vector the design
                point corresponding to f_vals.

            s_vals_mean (ndarray): A 1D array specifying the expected value
                of the simulation outputs for the x value being scalarized.

            s_vals_sd (ndarray): A 1D array specifying the standard deviation
                for each of the simulation outputs.

        Returns:
            float: The scalarized value.

        """

    def save(self, filename):
        """ Save important data from this class so that it can be reloaded.

        Note: If this function is left unimplemented, ParMOO will reinitialize
        a fresh instance after a save/load. If this is the desired behavior,
        then this method and the load method need not be implemented.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data should be saved.

        """

        raise NotImplementedError("This class method has not been implemented")

    def load(self, filename):
        """ Reload important data into this class after a previous save.

        Note: If this function is left unimplemented, ParMOO will reinitialize
        a fresh instance after a save/load. If this is the desired behavior,
        then this method and the save method need not be implemented.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data has been saved.

        """

        raise NotImplementedError("This class method has not been implemented")


class CompositeFunction(ABC):
    """ ABC defining ParMOO composite functions.

    Extend this class to create a callable object that matches ParMOO can
    use as a composite function, such as:
     - an Objective,
     - a Constraint, or
     - a Gradient.

    Contains 2 methods:
     * ``__init__(des_type, sim_type)``
     * ``__call__(x, sx)``

    The ``__init__`` method is already implemented, and is the constructor.
    It can be overwritten if additional inputs (besides the design variable
    and simulation output types) are needed.

    The ``__call__`` method is left to be implemented, and performs the
    composite function evaluation.

    """

    __slots__ = ['n', 'm', 'des_type', 'sim_type']

    def __init__(self, des_type, sim_type):
        """ Constructor for CompositeFunction class.

        Args:
            des_type (np.dtype): The numpy.dtype of the design variables.

            sim_type (np.dtype): The numpy.dtype of the simulation outputs.

        """

        # Try to read design variable type
        try:
            self.des_type = np.dtype(des_type)
        except TypeError:
            raise TypeError("des_type must contain a valid numpy.dtype")
        self.n = len(self.des_type.names)
        if self.n <= 0:
            raise ValueError("An illegal des_type was given")
        # Try to read simulation variable type
        try:
            self.sim_type = np.dtype(sim_type)
        except TypeError:
            raise TypeError("sim_type must contain a valid numpy.dtype")
        self.m = 0
        for name in self.sim_type.names:
            self.m += np.maximum(np.sum(self.sim_type[name].shape), 1)
        if self.m <= 0:
            raise ValueError("An illegal sim_type was given")
        return

    @abstractmethod
    def __call__(self, x, sx):
        """ Make CompositeFunction objects callable.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            float: The output of this objective for the input x (when
            defining objectives and constraint functions).

            OR

            dict, dict: Dictionaries with the same keys as x and sx, whose
            corresponding values contain the partials with respect to x and
            sx, respectively.

        """


class Embedder(ABC):
    """ ABC describing the embedding of design variables.

    This class contains the following methods:
     * ``getLatentDesTols()``
     * ``getFeatureDesTols()``
     * ``getEmbeddingSize()``
     * ``getInputType()``
     * ``getLowerBounds()``
     * ``getUpperBounds()``
     * ``embed(x)``
     * ``embed_grad(dx)``
     * ``extract(x)``

    """

    @abstractmethod
    def __init__(self, settings):
        """ Constructor for the Embedder class.

        Args:
            settings (dict): Contains any variable information that the user
                might need to provide.

        Returns:
            Embedder: A new Embedder object.

        """

    @abstractmethod
    def getLatentDesTols(self):
        """ Get the design tolerances along each dimension of the embedding.

        Returns:
            numpy.ndarray: array of design space tolerances after embedding.

        """

    @abstractmethod
    def getFeatureDesTols(self):
        """ Get the design tolerances in the feature space (pre-embedding).

        Returns:
            float: the design tolerance in the feature space -- a value of
            0 indicates that this is a discrete variable.

        """

    @abstractmethod
    def getEmbeddingSize(self):
        """ Get the dimension of the latent (embedded) space.

        Returns:
            int: the dimension of the latent space produced.

        """

    @abstractmethod
    def getInputType(self):
        """ Get the input type for this embedder.

        Note: Whatever the input type, the output type must always be a
        ndarray of one or more continuous variables in some range [lb, ub].

        Returns:
            str: A numpy string representation of the input type from the
            feature space.
            Currently supported values are: ["f8", "i4", "a25", or "u25"].

        """

    @abstractmethod
    def getLowerBounds(self):
        """ Get a vector of lower bounds for the embedded (latent) space.

        Returns:
            ndarray: A 1D array of lower bounds in embedded space whose size
                matches the output of ``getEmbeddingSize()``.

        """

    @abstractmethod
    def getUpperBounds(self):
        """ Get a vector of upper bounds for the embedded (latent) space.

        Returns:
            ndarray: A 1D array of upper bounds in embedded space whose size
                matches the output of ``getEmbeddingSize()``.

        """

    def embed(self, x):
        """ Embed a design input as an n-dimensional vector for ParMOO.

        Note: For best performance, make sure that jax can jit this method.

        Args:
            x (stype): The value of the design variable to embed, where
                stype matches the numpy-string type specified by
                getInputType().

        Returns:
            ndarray: A 1D array whose size matches the output of
            getEmbeddingSize() containing the embedding of x.

        """

        raise NotImplementedError("This Embedder has not implemented an "
                                  "embed method yet.")

    def embed_grad(self, dx):
        """ Embed a partial design gradient as a vector for ParMOO.

        Note: If not implemented, ParMOO will still work with gradient-free
        methods, but will not support autograd features.

        For best performance, make sure that jax can jit this method.

        Args:
            dx (float): The partial design gradient to embed.

        Returns:
            numpy.ndarray: A numpy array of length 1 containing a
            rescaling of x

        """

        raise NotImplementedError("This Embedder has not implemented an "
                                  "embed_grad method yet.")

    def extract(self, x):
        """ Extract a design input from an n-dimensional vector for ParMOO.

        Note: For best performance, make sure that jax can jit this method.

        Args:
            x (ndarray): A 1D array whose size matches the output of
                getEmbeddingSize() containing the embedding of x.

        Returns:
            stype: The value of the design variable to embed, where stype
            matches the numpy-string type specified by getInputType().

        """

        raise NotImplementedError("This Embedder has not implemented an "
                                  "extract method yet.")


class GlobalSearch(ABC):
    """ ABC describing global search techniques.

    This class contains the following methods.
     * ``startSearch(lb, ub)``
     * ``resumeSearch()``
     * ``save(filename)``
     * ``load(filename)``

    """

    @abstractmethod
    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the GlobalSearch class.

        Args:
            o (int): The number of objectives.

            lb (ndarray): A 1D array of lower bounds for the design space.

            ub (ndarray): A 1D array of upper bounds for the design space.

            hyperparams (dict): A dictionary of hyperparameters for the
                global search. It may contain any inputs specific to the
                search algorithm.

        Returns:
            GlobalSearch: A new GlobalSearch object.

        """

    @abstractmethod
    def startSearch(self, lb, ub):
        """ Begin a new global search.

        Args:
            lb (ndarray): A 1D array of lower bounds for the design space.

            ub (ndarray): A 1D array of upper bounds for the design space.

        Returns:
            ndarray: A 2D design matrix.

        """

    def resumeSearch(self):
        """ Resume a global search.

        Returns:
            ndarray: A 2D design matrix.

        """

        raise NotImplementedError("This class method has not been implemented")

    def save(self, filename):
        """ Save important data from this class so that it can be reloaded.

        Note: If this function is left unimplemented, ParMOO will reinitialize
        a fresh instance after a save/load. If this is the desired behavior,
        then this method and the load method need not be implemented.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data should be saved.

        """

        raise NotImplementedError("This class method has not been implemented")

    def load(self, filename):
        """ Reload important data into this class after a previous save.

        Note: If this function is left unimplemented, ParMOO will reinitialize
        a fresh instance after a save/load. If this is the desired behavior,
        then this method and the save method need not be implemented.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data has been saved.

        """

        raise NotImplementedError("This class method has not been implemented")


class SurrogateFunction(ABC):
    """ ABC describing surrogate functions.

    This class contains the following methods.
     * ``fit(x, f)``
     * ``update(x, f)``
     * ``setTrustRegion(center, radius)`` (default implementation provided)
     * ``evaluate(x)``
     * ``stdDev(x)``
     * ``improve(x, global_improv)`` (default implementation provided)
     * ``save(filename)``
     * ``load(filename)``

    """

    @abstractmethod
    def __init__(self, m, lb, ub, hyperparams):
        """ Constructor for the SurrogateFunction class.

        Args:
            m (int): The number of objectives to fit.

            lb (ndarray): A 1D array of lower bounds for the design space.

            ub (ndarray): A 1D array of upper bounds for the design space.

            hyperparams (dict): A dictionary of hyperparameters to be used
                by the surrogate models, including:
                 * des_tols (ndarray, optional): A 1D array whose length
                   matches lb and ub. Each entry is a number (greater than 0)
                   specifying the design space tolerance for that variable.

        Returns:
            SurrogateFunction: A new SurrogateFunction object.

        """

    @abstractmethod
    def fit(self, x, f):
        """ Fit a new surrogate to the given data.

        Args:
             x (ndarray): A 2D array containing the design points to fit.

             f (ndarray): A 2D array of the corresponding objectives values.

        """

    @abstractmethod
    def update(self, x, f):
        """ Update an existing surrogate model using new data.

        Args:
             x (ndarray): A 2D array containing new design points to fit.

             f (ndarray): A 2D array of the corresponding objectives values.

        """

    def setTrustRegion(self, center, radius):
        """ Alert the surrogate of the trust-region center and radius.

        Default implementation does nothing, which would be the case for a
        global surrogate model.

        Args:
            center (ndarray): A 1D array containing the center for a local fit.

            radius (ndarray or float): The radius for the local fit.

        """

        return

    @abstractmethod
    def evaluate(self, x):
        """ Evaluate the surrogate at a design point.

        Note: For best performance, make sure that jax can jit this method.

        Additionally, for compatibility with gradient-based solvers,
        this method must be implemented in jax and be differentiable
        via the jax.jacrev() tool.

        Args:
            x (ndarray): A 1D array containing the design point to evaluate.

        Returns:
            ndarray: A 1D array containing the predicted outputs at x.

        """

    def stdDev(self, x):
        """ Evaluate the standard deviation of the surrogate at x.

        Note: this method need not be implemented when the acquisition
        function does not use the model uncertainty.

        Additionally, for compatibility with gradient-based solvers,
        this method must be implemented in jax and be differentiable
        via the jax.jacrev() tool.

        Args:
            x (ndarray): A 1D array containing the design point to evaluate.

        Returns:
            ndarray: A 1D array containing the output standard deviation at x.

        """

        raise NotImplementedError("This class method has not been implemented")

    def improve(self, x, global_improv):
        """ Suggests a design to evaluate to improve the surrogate near x.

        A default implementation is given based on random sampling.
        Re-implement the improve method to overwrite the default
        policy.

        Args:
            x (ndarray): A 1D array containing a design point where greater
                accuracy is needed.

            global_improv (Boolean): When True, ignore the value of x and
                seek global model improvement.

        Returns:
            ndarray: A 2D array containing a list of (at least 1) design points
            that could be evaluated to improve the surrogate model's accuracy.

        """

        # Check that the x is legal
        try:
            if x.size != self.n:
                raise ValueError("x must have length n")
            elif (np.any(x < self.lb - self.eps) or
                  np.any(x > self.ub + self.eps)):
                raise ValueError("x cannot be infeasible")
        except AttributeError:
            raise TypeError("x must be a numpy array-like object")
        # Allocate the output array.
        x_new = np.zeros(self.n)
        if global_improv:
            # If global improvement has been specified, randomly select a
            # point from within the bound constraints.
            x_new[:] = self.lb[:] + (np.random.random(self.n)
                                     * (self.ub[:] - self.lb[:]))
            while any([np.all(np.abs(x_new - xj) < self.eps)
                       for xj in self.x_vals]):
                x_new[:] = self.lb[:] + (np.random.random(self.n)
                                         * (self.ub[:] - self.lb[:]))
        else:
            # Find the n+1 closest points to x in the current database
            diffs = np.asarray([np.abs(x - xj) / self.eps
                                for xj in self.x_vals])
            dists = np.asarray([np.amax(dj) for dj in diffs])
            inds = np.argsort(dists)
            diffs = diffs[inds]
            if dists[inds[self.n]] > 1.5:
                # Calculate the normalized sample standard dev along each axis
                stddev = np.asarray(tstd(diffs[:self.n+1], axis=0))
                stddev[:] = np.maximum(stddev, np.ones(self.n))
                stddev[:] = stddev[:] / np.amin(stddev)
                # Sample within B(x, dists[inds[self.n]] / stddev)
                rad = (dists[inds[self.n]] * self.eps) / stddev
                x_new = np.fmin(np.fmax(2.0 * (np.random.random(self.n) - 0.5)
                                        * rad[:] + x, self.lb), self.ub)
                while any([np.all(np.abs(x_new - xj) < self.eps)
                           for xj in self.x_vals]):
                    x_new = np.fmin(np.fmax(2.0 *
                                            (np.random.random(self.n) - 0.5)
                                            * rad[:] + x, self.lb), self.ub)
            else:
                # If the n+1st nearest point is too close, use global_improv.
                x_new[:] = self.lb[:] + np.random.random(self.n) \
                           * (self.ub[:] - self.lb[:])
                # If the nearest point is too close, resample.
                while any([np.all(np.abs(x_new - xj) < self.eps)
                           for xj in self.x_vals]):
                    x_new[:] = self.lb[:] + (np.random.random(self.n)
                                             * (self.ub[:] - self.lb[:]))
        # Return the point to be sampled in a 2d array.
        return np.asarray([x_new])

    def save(self, filename):
        """ Save important data from this class so that it can be reloaded.

        Note: If this function is left unimplemented, ParMOO will reinitialize
        a fresh instance after a save/load. If this is the desired behavior,
        then this method and the load method need not be implemented.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data should be saved.

        """

        raise NotImplementedError("This class method has not been implemented")

    def load(self, filename):
        """ Reload important data into this class after a previous save.

        Note: If this function is left unimplemented, ParMOO will reinitialize
        a fresh instance after a save/load. If this is the desired behavior,
        then this method and the save method need not be implemented.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data has been saved.

        """

        raise NotImplementedError("This class method has not been implemented")


class SurrogateOptimizer(ABC):
    """ ABC describing surrogate optimization techniques.

    This class contains the following methods.
     * ``setObjective(obj_func)`` (default implementation provided)
     * ``setSimulation(sim_func, sd_func)`` (default implementation provided)
     * ``setConstraints(constraint_func)`` (default implementation provided)
     * ``setPenalty(penaltyFunc, gradFunc)`` (default implementation provided)
     * ``setTrFunc(trFunc)`` (default implementation provided)
     * ``addAcquisition(*args)`` (default implementation provided)
     * ``returnResults(x, fx, sx, sdx)``
     * ``solve(x)``
     * ``save(filename)``
     * ``load(filename)``

    """

    @abstractmethod
    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the SurrogateOptimizer class.

        Args:
            o (int): The number of objectives.

            lb (ndarray): A 1D array of lower bounds for the design space.

            ub (ndarray): A 1D array of upper bounds for the design space.

            hyperparams (dict): A dictionary of hyperparameters for the
                optimization procedure.

        Returns:
            SurrogateOptimizer: A new SurrogateOptimizer object.

        """

    def setObjective(self, obj_func):
        """ Add a vector-valued objective function that will be solved.

        Args:
            obj_func (function): A vector-valued function that can be evaluated
                to solve the surrogate optimization problem.

        """

        # Check whether obj_func() has an appropriate signature
        if callable(obj_func):
            if len(inspect.signature(obj_func).parameters) != 2:
                raise ValueError("obj_func() must accept exactly two inputs")
            else:
                # Add obj_func to the problem
                self.objectives = obj_func
        else:
            raise TypeError("obj_func() must be callable")
        return

    def setSimulation(self, sim_func, sd_func):
        """ Add a vector-valued simulation function, used to calculate objs.

        Args:
            sim_func (function): A vector-valued function that can be evaluated
                to determine the surrogate-predicted simulation outputs.

            sd_func (function): A vector-valued function that can be evaluated
                to determine the standard deviations of the surrogate
                predictions.

        """

        # Check whether sim_func() has an appropriate signature
        if callable(sim_func):
            if len(inspect.signature(sim_func).parameters) != 1:
                raise ValueError("sim_func() must accept exactly one input")
            else:
                # Add sim_func to the problem
                self.simulations = sim_func
        else:
            raise TypeError("sim_func() must be callable")
        # Check whether sd_func() has an appropriate signature
        if callable(sd_func):
            if len(inspect.signature(sd_func).parameters) != 1:
                raise ValueError("sd_func() must accept exactly one input")
            else:
                self.sim_sd = sd_func
        else:
            raise TypeError("sd_func() must be callable")
        return

    def setPenalty(self, penalty_func):
        """ Add a matrix-valued gradient function for obj_func.

        Args:
            penalty_func (function): A vector-valued penalized objective
                that incorporates a penalty for violating constraints.

            grad_func (function): A matrix-valued function that can be
                evaluated to obtain the Jacobian matrix for obj_func.

        """

        # Check whether penalty_func() has an appropriate signature
        if callable(penalty_func):
            if len(inspect.signature(penalty_func).parameters) != 2:
                raise ValueError("penalty_func must accept exactly two inputs")
            else:
                # Add penalty to the problem
                self.penalty_func = penalty_func
        else:
            raise TypeError("penalty_func must be callable")
        return

    def setConstraints(self, constraint_func):
        """ Add a constraint function that will be satisfied.

        Args:
            constraint_func (function): A vector-valued function from the
                design space whose components correspond to constraint
                violations. If the problem has only bound constraints, this
                function returns zeros.

        """

        # Check whether constraint_func() has an appropriate signature
        if callable(constraint_func):
            if len(inspect.signature(constraint_func).parameters) != 2:
                raise ValueError("constraint_func() must accept exactly two"
                                 + " input")
            else:
                # Add constraint_func to the problem
                self.constraints = constraint_func
        else:
            raise TypeError("constraint_func() must be callable")
        return

    def setTrFunc(self, trFunc):
        """ Add a TR setter function for alerting surrogates.

        Args:
            trFunc (function): A function with 2 inputs, which the optimizer
                must call prior to solving each surrogate optimization problem
                in order to set the trust-region center and radius.

        """

        # Check whether trFunc() has an appropriate signature
        if callable(trFunc):
            if len(inspect.signature(trFunc).parameters) != 2:
                raise ValueError("trFunc() must accept exactly 2 inputs")
            else:
                # Add obj_func to the problem
                self.setTR = trFunc
        else:
            raise TypeError("trFunc() must be callable")
        return

    def returnResults(self, x, fx, sx, sdx):
        """ This is a callback function to collect evaluation results.

        Implement this function to receive the results of each
        true simulation evaluation from the MOOP class at runtime.

        Args:
            x (ndarray): A 1D array with the design point evaluated.

            fx (ndarray): A 1D array with the objective function values at x.

            sx (ndarray): The simulation function values at x.

            sdx (ndarray): The standard deviation in the simulation prediction.

        """

        return

    def addAcquisition(self, *args):
        """ Add an acquisition function for the surrogate optimizer.

        Args:
            *args (AcquisitionFunction): Acquisition functions that are used
                to scalarize the list of objectives in order to solve the
                surrogate optimization problem.

        """

        # Check for illegal inputs
        if not all([isinstance(arg, AcquisitionFunction) for arg in args]):
            raise TypeError("Args must be instances of AcquisitionFunction")
        # Append all arguments to the acquisitions list
        for arg in args:
            self.acquisitions.append(arg)
        return

    @abstractmethod
    def solve(self, x_k):
        """ Solve the surrogate problem.

        You may assume that the following internal attributes are defined
        and contain callable definitions of the objective, constraint,
        penalty, and simulation (surrogate) functions, respectively:
         * ``self.objectives``,
         * ``self.constraints``,
         * ``self.penalty_func``, and
         * ``self.simulations``.

        Additionally, you may assume that:
         * ``self.acquisitions`` contains a list of one or more
           ``AcqusitionFunction`` object instances, each of whose
           ``acq.scalarize(f_vals, x_vals, s_vals_mean, s_vals_sd)``
           is set and ready to call; and
         * ``self.setTR(x, r)`` can be called to set a trust-region
           centered at ``x`` with radius ``r`` (and re-fit the surrogates
           accordingly).

        Note: If implementing your own solver, try to jit (or re-jit) any of
        the objective, constraint, penalty, simulation surrogate, and/or
        acquisition functions after each call to ``self.setTR``.
        Additionally, if provided by the user,
        the objectives, constraints, penalty, and acq.scalarize,
        functions should all be differentiable by importing and
        calling ``jax.jacrev()``.

        Args:
            x_k (ndarray): A 2D array containing a list of current iterates.

        Returns:
            ndarray: A 2D array matching the shape of x_k specifying x_{k+1}.

        """

    def save(self, filename):
        """ Save important data from this class so that it can be reloaded.

        Note: If this function is left unimplemented, ParMOO will reinitialize
        a fresh instance after a save/load. If this is the desired behavior,
        then this method and the load method need not be implemented.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data should be saved.

        """

        raise NotImplementedError("This class method has not been implemented")

    def load(self, filename):
        """ Reload important data into this class after a previous save.

        Note: If this function is left unimplemented, ParMOO will reinitialize
        a fresh instance after a save/load. If this is the desired behavior,
        then this method and the save method need not be implemented.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data has been saved.

        """

        raise NotImplementedError("This class method has not been implemented")

""" Abstract base class (ABC) for ParMOO's SimulationDatabase structure. """

from abc import ABC, abstractmethod


class SimulationDatabase(ABC):
    """ ABC database specialized for multiobjective simulation data.

    Class contains the following adders/getters:
     * ``SimulationDatabase.addDesign(*args)``
     * ``SimulationDatabase.addSimulation(*args)``
     * ``SimulationDatabase.addObjective(*args)``
     * ``SimulationDatabase.addConstraint(*args)``
     * ``SimulationDatabase.getDesignType()``
     * ``SimulationDatabase.getSimulationType()``
     * ``SimulationDatabase.getObjectiveType()``
     * ``SimulationDatabase.getConstraintType()``
     * ``SimulationDatabase.startDatabase()``
     * ``SimulationDatabase.checkSimDb(x, sim_name)``
     * ``SimulationDatabase.checkObjDb(x)``
     * ``SimulationDatabase.updateSimDb(x, sx, sim_name)``
     * ``SimulationDatabase.updateObjDb(x, fx, cx)``
     * ``SimulationDatabase.isEmpty()``
     * ``SimulationDatabase.browseCompleteSimulations()``
     * ``SimulationDatabase.getPF(format='ndarray')``
     * ``SimulationDatabase.getSimulationData(format='ndarray')``
     * ``SimulationDatabase.getNewSimulationData()``
     * ``SimulationDatabase.getObjectiveData(format='ndarray')``
     * ``SimulationDatabase.setCheckpoint(checkpoint, filename="parmoo")``
     * ``SimulationDatabase.checkpointSimData(x, sx, sim_name, filename)``
     * ``SimulationDatabase.checkpointObjData(x, fx, cx, filename)``
     * ``SimulationDatabase.loadCheckpoint(filename="parmoo")``

    """

    def __init__(self, hyperparams):
        """ Initializer for the SimulationDatabase class.

        Args:
            hyperparams (dict): Any parameters for configuring the database.

        """

        pass

    @abstractmethod
    def addDesign(self, name, dtype, tolerance):
        """ Add a new design variable to the SimulationDatabase schema.

        Args:
            name (str, optional): The unique name of this design variable.
            dtype (str): The string-representation for the numpy dtype for this
                design variable.
            tolerance (float): The tolerance up to which two different values
                for this design variables should be considered as "the same."
                If a zero is given, then only exact equality is checked.

        """

    @abstractmethod
    def addSimulation(self, name, m):
        """ Add new simulations to the SimulationDatabase schema.

        Args:
            name (str): The unique name of this simulation output.
            m (int): The number of outputs for this simulation.

        """

    @abstractmethod
    def addObjective(self, name):
        """ Add a new objective to the SimulationDatabase schema.

        Args:
            name (str): The unique name of this objective output.

        """

    @abstractmethod
    def addConstraint(self, name):
        """ Add a new constraint to the SimulationDatabase schema.

        Args:
            name (str, optional): The unique name of this constraint violation.

        """

    @abstractmethod
    def getDesignType(self):
        """ Get the numpy dtype of all design points for this MOOP.

        Returns:
            dtype: The numpy dtype of this MOOP's design points.
            If no design variables have yet been added, returns None.

        """

    @abstractmethod
    def getSimulationType(self):
        """ Get the numpy dtypes of the simulation outputs for this MOOP.

        Returns:
            dtype: The numpy dtype of this MOOP's simulation outputs.
            If no simulations have been given, returns None.

        """

    @abstractmethod
    def getObjectiveType(self):
        """ Get the numpy dtype of an objective point for this MOOP.

        Returns:
            dtype: The numpy dtype of this MOOP's objective points.
            If no objectives have yet been added, returns None.

        """

    @abstractmethod
    def getConstraintType(self):
        """ Get the numpy dtype of the constraint violations for this MOOP.

        Returns:
            dtype: The numpy dtype of this MOOP's constraint violation
            outputs. If no constraint functions have been given, returns None.

        """

    @abstractmethod
    def startDatabase(self):
        """ Initialize the SimulationDatabase. """

    @abstractmethod
    def checkSimDb(self, x, sim_name):
        """ Check self.sim_db[sim_name] to see if the design x was evaluated.

        Args:
            x (dict): A Python dictionary specifying the keys/names and
                corresponding values of a design point to search for.
            sim_name (str): The name of the simulation whose database will be
                searched.

        Returns:
            None or numpy.ndarray: returns None if x is not in the database
            for simulation "sim_name" (up to the design tolerance). Otherwise,
            returns the corresponding value of sx.

        """

    @abstractmethod
    def checkObjDb(self, x):
        """ Check self.obj_db to see if the design x was evaluated.

        Args:
            x (dict): A Python dictionary specifying the keys/names and
                corresponding values of a design point to search for.

        Returns:
            None or pair of numpy.ndarrays: returns None if x is not in the
            database (up to the design tolerance). Otherwise, returns the
            corresponding value of (fx, cx) where fx is the vector-valued
            objective and cx is the vector-valued constraint violation at x.

        """

    @abstractmethod
    def updateSimDb(self, x, sx, sim_name):
        """ Update sim_db[sim_name] by adding a design/simulation output pair.

        Args:
            x (dict): A Python dictionary specifying the keys/names and
                corresponding values of a design point to add.

            sx (ndarray): A 1D array containing the corresponding
                simulation output(s).

            sim_name (str): The name of the simulation to whose database the
                pair (x, sx) will be added into.

        """

    @abstractmethod
    def updateObjDb(self, x, fx, cx):
        """ Update the internal objective database with a true evaluation of x.

        Args:
            x (dict): A Python dictionary containing the value of the design
                variable to add to ParMOO's database.
            fx (numpy.ndarray): An array of objective values to add.
            cx (numpy.ndarray): An array of constraint values to add.

        """

    @abstractmethod
    def isEmpty(self):
        """ Check whether the database is completely empty.

        Returns:
            bool: True if and only if every simulation database and the
            objective database is completely empty (size 0).

        """

    @abstractmethod
    def browseCompleteSimulations(self):
        """ Browse all design values that are present in every sim database.

        Yields:
            A sequence of tuples (x, sx) where each x is a (dict) design point
            that is present in every internal simulation database, and sx is a
            dictionary of simulation outputs from each of these database.

        """

    @abstractmethod
    def getPF(self, format='ndarray'):
        """ Extract nondominated and efficient sets from internal databases.

        Args:
            format (str, optional): Either 'ndarray' (default) or 'pandas',
                in order to produce output as a numpy structured array or
                pandas dataframe. Note: format='pandas' is only valid for
                named inputs.

        Returns:
            numpy structured array or pandas DataFrame: Either a structured
            array or dataframe (depending on the option selected above)
            whose column/key names match the names of the design variables,
            objectives, and constraints. It contains a discrete approximation
            of the Pareto front and efficient set.

        """

    @abstractmethod
    def getSimulationData(self, format='ndarray'):
        """ Extract all computed simulation outputs from the MOOP's database.

        Args:
            format (str, optional): Either 'ndarray' (default) or 'pandas',
                in order to produce output as a numpy structured array or
                pandas dataframe. Note: format='pandas' is only valid for
                named inputs.

        Returns:
            dict: A Python dictionary whose keys match the names of the
            simulations. Each value is either a numpy structured array or
            pandas dataframe (depending on the option selected above)
            whose column/key names match the names of the design variables
            plus either and 'out' field for single-output simulations,
            or 'out_1', 'out_2', ... for multi-output simulations.

        """

    @abstractmethod
    def getNewSimulationData(self):
        """ Extract simulation outputs that have not yet been viewed.

        Returns:
            dict: A Python dictionary whose keys match the names of the
            simulations and whose values are the new data for each
            variable/simulation output.

        """

    @abstractmethod
    def getObjectiveData(self, format='ndarray'):
        """ Extract all computed objective scores from this database.

        Args:
            format (str, optional): Either 'ndarray' (default) or 'pandas',
                in order to produce output as a numpy structured array or
                pandas dataframe. Note: format='pandas' is only valid for
                named inputs.

        Returns:
            numpy structured array or pandas DataFrame: Either a structured
            array or dataframe (depending on the option selected above)
            whose column/key names match the names of the design variables,
            objectives, and constraints. It contains the results for every
            fully evaluated design point.

        """

    @abstractmethod
    def setCheckpoint(self, checkpoint, filename="parmoo"):
        """ Activate checkpointing.

        Args:
            checkpoint (bool): Turn checkpointing on (True) or off (False).
            filename (str, optional): Set the base checkpoint filename/path.
                The checkpoint file will have the JSON format and the
                extension ".simdb.json" appended to the end of filename.

        """

    @abstractmethod
    def checkpointSimData(self, x, sx, sim_name, filename="parmoo"):
        """ Append the given simulation data point to the checkpoint file.

        Args:
            x (dict or numpy structured element): The design value to append.
            sx (dict or numpy structured element): The simulation output to
                append.
            sim_name (str): The simulation name/index to append to.
            filename (str, optional): The filepath to the checkpointing
                file(s). Do not include file extensions, they will be
                appended automatically. Defaults to the value "parmoo"
                (filename will be "parmoo.simdb.json").

        """

    @abstractmethod
    def checkpointObjData(self, x, fx, cx, filename="parmoo"):
        """ Append the given objective data point to the checkpoint file.

        Args:
            x (dict or numpy structured element): The design value to append.
            fx (dict or numpy structured element): The objective values to
                append.
            cx (dict or numpy structured element): The constraint violations to
                append.
            filename (str, optional): The filepath to the checkpointing
                file(s). Do not include file extensions, they will be
                appended automatically. Defaults to the value "parmoo"
                (filename will be "parmoo.simdb.json").

        """

    @abstractmethod
    def loadCheckpoint(self, filename="parmoo"):
        """ Reload from the given checkpoint file.

        Args:
            filename (str, optional): The filepath to the checkpointing
                file(s). Do not include file extensions, they will be
                appended automatically. Defaults to the value "parmoo"
                (filename will be "parmoo.simdb.json").

        """

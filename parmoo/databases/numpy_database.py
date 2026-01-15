
""" Contains the NumpyDatabase class for storing simulation results.

``parmoo.NumpyDatabase`` is the base class for storing multiobjective
simulation outputs. Each NumpyDatabase object may contain several
simulations, and their corresponding objective and constraint violation scores.

"""

import json
import logging
import numpy as np
from os.path import exists as file_exists
import pandas as pd

from parmoo.databases.simulation_database import SimulationDatabase
from parmoo.utilities.error_checks import check_names
from parmoo.utilities.moop_utils import approx_equal


class NumpyDatabase(SimulationDatabase):
    """ A Database class specialized for multiobjective data.

    To define the NumpyDatabase, add each design variable, simulation,
    objective, and constraint by using the following functions:
     * ``NumpyDatabase.addDesign(*args)``
     * ``NumpyDatabase.addSimulation(*args)``
     * ``NumpyDatabase.addObjective(*args)``
     * ``NumpyDatabase.addConstraint(*args)``

    After creating a NumpyDatabase, the following methods may be useful
    for getting the numpy.dtype of the input/output arrays:
     * ``NumpyDatabase.getDesignType()``
     * ``NumpyDatabase.getSimulationType()``
     * ``NumpyDatabase.getObjectiveType()``
     * ``NumpyDatabase.getConstraintType()``

    To check or add to the simulation database, use:
     * ``NumpyDatabase.checkSimDb(x, sim_name)``
     * ``NumpyDatabase.updateSimDb(x, sx, sim_name)``

    To check or add to the objective database, use:
     * ``NumpyDatabase.checkObjDb(x)``
     * ``NumpyDatabase.updateObjDb(x, fx, cx)``

    Finally, the following methods are used to retrieve (filtered) simulation
    and objective data:
     * ``NumpyDatabase.isEmpty()``
     * ``NumpyDatabase.browseCompleteSimulations()``
     * ``NumpyDatabase.getPF(format='ndarray')``
     * ``NumpyDatabase.getSimulationData(format='ndarray')``
     * ``NumpyDatabase.getNewSimulationData()``
     * ``NumpyDatabase.getObjectiveData(format='ndarray')``

    To activate checkpointing, use:
     * ``NumpyDatabase.setCheckpoint(checkpoint, filename="parmoo")``

    Then to force a save or load of the current state, use:
     * ``NumpyDatabase.checkpointSimData(x, sx, sim_name, filename="parmoo")``
     * ``NumpyDatabase.checkpointObjData(x, fx, cx, filename="parmoo")``
     * ``NumpyDatabase.loadCheckpoint(filename="parmoo")``

    """

    __slots__ = [
        # Schemas
        'des_schema', 'sim_schema', 'obj_schema', 'con_schema',
        # Design tolerances for lookup
        'des_tols',
        # Compiled flag
        'running',
        # Checkpointing markers
        'checkpoint_data', 'checkpoint_file', 'checkpoint_new',
        # Database information
        'obj_db', 'sim_db',
    ]

    def __init__(self, hyperparams):
        """ Initializer for the NumpyDatabase class.

        Args:
            hyperparams (dict): Any parameters for configuring the database.

        """

        # Initialize the schemas
        self.des_schema, self.sim_schema = [], []
        self.obj_schema, self.con_schema = [], []
        # Initialize design tolerances for lookup
        self.des_tols = {}
        # Initialize the running flag
        self.running = False
        # Initialize checkpointing markers
        self.checkpoint_data = False
        self.checkpoint_file = "parmoo"
        self.checkpoint_new = True
        # Initialize the database
        self.obj_db, self.sim_db = None, None

    def addDesign(self, name, dtype, tolerance):
        """ Add a new design variable to the NumpyDatabase schema.

        Args:
            name (str): The unique name of this design variable.
            dtype (str): The string-representation for the numpy dtype for this
                design variable.
            tolerance (float): The tolerance up to which two different values
                for this design variables should be considered as "the same."
                If a zero is given, then only exact equality is checked.

        """

        check_names(
            name,
            self.des_schema, self.sim_schema, self.obj_schema, self.con_schema
        )
        self.des_schema.append((name, dtype))
        self.des_tols[name] = tolerance

    def addSimulation(self, name, m):
        """ Add new simulations to the NumpyDatabase schema.

        Args:
            name (str): The unique name of this simulation output.
            m (int): The number of outputs for this simulation.

        """

        check_names(
            name,
            self.des_schema, self.sim_schema, self.obj_schema, self.con_schema
        )
        if m > 1:
            self.sim_schema.append((name, 'f8', m))
        else:
            self.sim_schema.append((name, 'f8'))

    def addObjective(self, name):
        """ Add a new objective to the NumpyDatabase schema.

        Args:
            name (str): The unique name of this objective output.

        """

        check_names(
            name,
            self.des_schema, self.sim_schema, self.obj_schema, self.con_schema
        )
        self.obj_schema.append((name, 'f8'))

    def addConstraint(self, name):
        """ Add a new constraint to the NumpyDatabase schema.

        Args:
            name (str, optional): The unique name of this constraint violation.

        """

        check_names(
            name,
            self.des_schema, self.sim_schema, self.obj_schema, self.con_schema
        )
        self.con_schema.append((name, 'f8'))

    def getDesignType(self):
        """ Get the numpy dtype of all design points for this MOOP.

        Returns:
            dtype: The numpy dtype of this MOOP's design points.
            If no design variables have yet been added, returns None.

        """

        if len(self.des_schema) < 1:
            return None
        else:
            return np.dtype(self.des_schema)

    def getSimulationType(self):
        """ Get the numpy dtypes of the simulation outputs for this MOOP.

        Returns:
            dtype: The numpy dtype of this MOOP's simulation outputs.
            If no simulations have been given, returns None.

        """

        if len(self.sim_schema) < 1:
            return None
        else:
            return np.dtype(self.sim_schema)

    def getObjectiveType(self):
        """ Get the numpy dtype of an objective point for this MOOP.

        Returns:
            dtype: The numpy dtype of this MOOP's objective points.
            If no objectives have yet been added, returns None.

        """

        if len(self.obj_schema) < 1:
            return None
        else:
            return np.dtype(self.obj_schema)

    def getConstraintType(self):
        """ Get the numpy dtype of the constraint violations for this MOOP.

        Returns:
            dtype: The numpy dtype of this MOOP's constraint violation
            outputs. If no constraint functions have been given, returns None.

        """

        if len(self.con_schema) < 1:
            return None
        else:
            return np.dtype(self.con_schema)

    def startDatabase(self):
        """ Initialize the NumpyDatabase. """

        # For safety reasons, don't let silly users delete their data
        if not self.isEmpty():
            raise RuntimeError(
                "Cannot re-compile a MOOP with a nonempty database. "
                "If that's really what you want, then please reset this MOOP."
            )
        logging.info("   Initializing ParMOO's internal databases...")
        self.obj_db = {
            'x_vals': np.zeros(50, dtype=self.des_schema),
            'f_vals': np.zeros(50, dtype=self.obj_schema),
            'c_vals': np.zeros(50, dtype=self.con_schema),
            'n': 0,
        }
        self.sim_db = {}
        for stype in self.sim_schema:
            if len(stype) > 2:
                mi = stype[2]
            else:
                mi = 1
            self.sim_db[stype[0]] = {
                'x_vals': np.zeros(50, dtype=self.des_schema),
                's_vals': np.zeros((50, mi)),
                'n': 0,
                'n_old': 0,
            }
        self.running = True
        logging.info("   Done.")
        # If checkpointing is on, we need to start the checkpoint file
        if self.checkpoint_data:
            self._checkpoint_metadata(self.checkpoint_file)

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

        if not self.running:
            raise RuntimeError("Cannot check a database that is not running")
        if sim_name not in self.sim_db:
            raise ValueError(f"{sim_name} is not a legal name/index")
        for i in range(self.sim_db[sim_name]['n']):
            if approx_equal(
                x, self.sim_db[sim_name]['x_vals'][i], self.des_tols
            ):
                return self.sim_db[sim_name]['s_vals'][i]
        return None

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

        if not self.running:
            raise RuntimeError("Cannot check a database that is not running")
        for i in range(self.obj_db['n']):
            if approx_equal(
                x, self.obj_db['x_vals'][i], self.des_tols
            ):
                return self.obj_db['f_vals'][i], self.obj_db['c_vals'][i]
        return None

    def updateSimDb(self, x, sx, sim_name):
        """ Update sim_db[sim_name] by adding a design/simulation output pair.

        Args:
            x (dict): A Python dictionary specifying the keys/names and
                corresponding values of a design point to add.

            sx (ndarray or list): A 1D array containing the corresponding
                simulation output(s).

            sim_name (str): The name of the simulation to whose database the
                pair (x, sx) will be added into.

        """

        if not self.running:
            raise RuntimeError("Cannot add to a database that is not running")
        if sim_name not in self.sim_db:
            raise ValueError(f"{sim_name} is not a legal name/index")
        # Convert all simulation data to flat arrays
        sx_flat = np.array(sx).flatten()
        i = self.sim_db[sim_name]['n']
        # Check if database needs to be resized
        if i >= len(self.sim_db[sim_name]['x_vals']):
            self.sim_db[sim_name]['x_vals'] = np.append(
                self.sim_db[sim_name]['x_vals'],
                np.zeros(i, dtype=self.des_schema), axis=0
            )
            self.sim_db[sim_name]['s_vals'] = np.append(
                self.sim_db[sim_name]['s_vals'],
                np.zeros((i, sx_flat.size)), axis=0
            )
        for key in self.des_schema:
            self.sim_db[sim_name]['x_vals'][key[0]][i] = x[key[0]]
        self.sim_db[sim_name]['s_vals'][i, :] = sx_flat[:]
        self.sim_db[sim_name]['n'] += 1
        # If various checkpointing modes are on, then save the current states
        if self.checkpoint_data:
            self.checkpointSimData(
                x, sx_flat, sim_name, filename=self.checkpoint_file
            )

    def updateObjDb(self, x, fx, cx):
        """ Update the internal objective database with a true evaluation of x.

        Args:
            x (dict): A dictionary containing the value of the design variable
                to add to ParMOO's database.
            fx (dict): A dictionary containing the values of the corresponding
                objective scores to add to ParMOO's database.
            cx (dict): A dictionary containing the values of the corresponding
                constraint violations to add to ParMOO's database.

        """

        if not self.running:
            raise RuntimeError("Cannot add to a database that is not running")
        # Resize the database if needed
        i = self.obj_db['n']
        if i >= len(self.obj_db['x_vals']):
            self.obj_db['x_vals'] = np.append(
                self.obj_db['x_vals'], np.zeros(i, dtype=self.des_schema),
                axis=0
            )
            self.obj_db['f_vals'] = np.append(
                self.obj_db['f_vals'], np.zeros(i, dtype=self.obj_schema),
                axis=0
            )
            self.obj_db['c_vals'] = np.append(
                self.obj_db['c_vals'], np.zeros(i, self.con_schema),
                axis=0
            )
        for key in self.des_schema:
            self.obj_db['x_vals'][key[0]][i:i+1] = x[key[0]]
        for key in self.obj_schema:
            self.obj_db['f_vals'][key[0]][i:i+1] = fx[key[0]]
        for key in self.con_schema:
            self.obj_db['c_vals'][key[0]][i:i+1] = cx[key[0]]
        self.obj_db['n'] += 1
        # If various checkpointing modes are on, then save the current states
        if self.checkpoint_data:
            self.checkpointObjData(x, fx, cx, filename=self.checkpoint_file)

    def isEmpty(self):
        """ Check whether the database is completely empty.

        Returns:
            bool: True if and only if every simulation database and the
            objective database is completely empty (size 0).

        """

        return (
            (self.obj_db is None or self.obj_db['n'] == 0) and
            (
                self.sim_db is None or
                all([self.sim_db[key]['n'] == 0 for key in self.sim_db])
            )
        )

    def browseCompleteSimulations(self):
        """ Browse all design values that are present in every sim database.

        Yields:
            A sequence of tuples (x, sx) where each x is a (dict) design point
            that is present in every internal simulation database, and sx is a
            dictionary of simulation outputs from each of these database.

        """

        if not self.running:
            raise RuntimeError("Cannot browse a database that is not running")
        if len(self.sim_schema) > 0:
            sim0 = self.sim_schema[0]
            n0 = self.sim_db[sim0[0]]['n']
            for xi, sxi in zip(
                self.sim_db[sim0[0]]['x_vals'][:n0],
                self.sim_db[sim0[0]]['s_vals'][:n0]
            ):
                # Initialize the x vals and s vals
                x_vals = {}
                for name in self.des_schema:
                    x_vals[name[0]] = xi[name[0]]
                if len(sim0) > 2:
                    s_vals = {sim0[0]: sxi.copy()}
                else:
                    s_vals = {sim0[0]: sxi[0]}
                # Look for matches in all other simulation databases
                matched = True
                for simi in self.sim_schema[1:]:
                    matched = False
                    ni = self.sim_db[simi[0]]['n']
                    for xj, sxj in zip(
                        self.sim_db[simi[0]]['x_vals'][:ni],
                        self.sim_db[simi[0]]['s_vals'][:ni]
                    ):
                        if approx_equal(x_vals, xj, self.des_tols):
                            if len(simi) > 2:
                                s_vals[simi[0]] = sxj.copy()
                            else:
                                s_vals[simi[0]] = sxj[0]
                            matched = True
                            break  # Break once we found a match
                    if not matched:
                        break  # Break if there was no match
                if matched:
                    yield x_vals, s_vals

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

        if not self.running:
            raise RuntimeError("Cannot get a database that is not running")
        n = self.obj_db['n']
        o = len(self.obj_schema)
        p = len(self.con_schema)
        dt = self.obj_schema[0][1]
        # Create a view of the objective and constraint values for computation
        f_view = self.obj_db['f_vals'][:n].view(dt).reshape(-1, o)
        if p > 0:
            c_view = self.obj_db['c_vals'][:n].view(dt).reshape(-1, p)
        # Initialize the output arrays
        ndpts = 0
        nondom_out = {
            'x_vals': np.zeros(n, dtype=self.des_schema),
            'f_vals': np.zeros(n, dtype=self.obj_schema),
            'c_vals': np.zeros(n, dtype=self.con_schema)
        }
        # Create a view of the output array for easy computations
        nondom_view = nondom_out['f_vals'].view(dt).reshape(n, o)
        # Loop over the f-values in lexicographical order
        lex_inds = np.lexsort(f_view.T)
        for i in lex_inds:
            if (
                (p == 0 or np.all(c_view[i, :] < 1e-8)) and
                np.all(np.any(
                    f_view[i, :] < nondom_view[:ndpts, :], axis=1
                ))
            ):
                nondom_out['x_vals'][ndpts] = self.obj_db['x_vals'][i]
                nondom_out['f_vals'][ndpts] = self.obj_db['f_vals'][i]
                nondom_out['c_vals'][ndpts] = self.obj_db['c_vals'][i]
                ndpts += 1
        # Extract the results
        result = np.zeros(
            ndpts, dtype=(self.des_schema + self.obj_schema + self.con_schema)
        )
        for dt in self.des_schema:
            result[dt[0]] = nondom_out['x_vals'][dt[0]][:ndpts]
        for dt in self.obj_schema:
            result[dt[0]] = nondom_out['f_vals'][dt[0]][:ndpts]
        for dt in self.con_schema:
            result[dt[0]] = nondom_out['c_vals'][dt[0]][:ndpts]
        if format == 'pandas':
            return pd.DataFrame(result)
        elif format == 'ndarray':
            return result
        else:
            raise ValueError(f"{format} is an invalid value for 'format'")

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

        if not self.running:
            raise RuntimeError("Cannot get a database that is not running")
        # Build a results dict with a key for each simulation
        result = {}
        for i, sname in enumerate(self.sim_schema):
            # Construct the dtype for this simulation database
            dt = self.des_schema.copy()
            if len(sname) == 2:
                dt.append(('out', sname[1]))
            else:
                dt.append(('out', sname[1], sname[2]))
            # Fill the results array
            n = self.sim_db[sname[0]]['n']
            result[sname[0]] = np.zeros(n, dtype=dt)
            for j, xj in enumerate(self.sim_db[sname[0]]['x_vals'][:n]):
                for (name, t) in self.des_schema:
                    result[sname[0]][name][j] = xj[name]
            if len(sname) > 2:
                result[sname[0]]['out'] = self.sim_db[sname[0]]['s_vals'][:n]
            else:
                result[sname[0]]['out'] = \
                    self.sim_db[sname[0]]['s_vals'][:n, 0]
        if format == 'pandas':
            # For simulation data, converting to pandas is a little more
            # complicated...
            result_pd = {}
            for i, snamei in enumerate(result.keys()):
                rtempi = {}
                for (name, t) in self.des_schema:
                    rtempi[name] = result[snamei][name]
                # Need to break apart the output column manually
                if len(self.sim_schema[i]) > 2:
                    for j in range(self.sim_schema[i][2]):
                        rtempi[f'out_{j}'] = result[snamei]['out'][:, j]
                else:
                    rtempi['out'] = result[snamei]['out'][:]
                # Create dictionary of dataframes, indexed by sim names
                result_pd[snamei] = pd.DataFrame(rtempi)
            return result_pd
        elif format == 'ndarray':
            return result
        else:
            raise ValueError(f"{format} is an invalid value for 'format'")

    def getNewSimulationData(self):
        """ Extract simulation outputs that have not yet been viewed.

        Returns:
            dict: A Python dictionary whose keys match the names of the
            simulations and whose values are the new data for each
            variable/simulation output.

        """

        if not self.running:
            raise RuntimeError("Cannot get a database that is not running")
        # Build a results dict with a key for each simulation
        result = {}
        for sname in self.sim_schema:
            # Construct the dtype for this simulation database
            dt = self.des_schema.copy()
            if len(sname) > 2:
                dt.append(('out', sname[1], sname[2]))
            else:
                dt.append(('out', sname[1]))
            # Fill the results arrays with entries n_old:n
            result[sname[0]] = np.zeros(
                self.sim_db[sname[0]]['n'] - self.sim_db[sname[0]]['n_old'],
                dtype=dt
            )
            n_old = self.sim_db[sname[0]]['n_old']
            n = self.sim_db[sname[0]]['n']
            for j in range(n_old, n):
                for (name, t) in self.des_schema:
                    result[sname[0]][name][j - n_old] = \
                        self.sim_db[sname[0]]['x_vals'][name][j]
            if len(sname) > 2:
                result[sname[0]]['out'] = \
                    self.sim_db[sname[0]]['s_vals'][n_old:n]
            else:
                result[sname[0]]['out'] = \
                    self.sim_db[sname[0]]['s_vals'][n_old:n, 0]
            # Update the tracker
            self.sim_db[sname[0]]['n_old'] = n
        if self.checkpoint_data:
            self._log_new_data_call(self.checkpoint_file)
        return result

    def getObjectiveData(self, format='ndarray'):
        """ Extract all computed objective scores from this MOOP's database.

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

        if not self.running:
            raise RuntimeError("Cannot get a database that is not running")
        # Initialize result array
        n = self.obj_db['n']
        result = np.zeros(
            n, dtype=(self.des_schema + self.obj_schema + self.con_schema)
        )
        # Extract all results
        for (name, t) in self.des_schema:
            result[name][:] = self.obj_db['x_vals'][name][:n]
        for (name, t) in self.obj_schema:
            result[name][:] = self.obj_db['f_vals'][name][:n]
        for (name, t) in self.con_schema:
            result[name][:] = self.obj_db['c_vals'][name][:n]
        if format == 'pandas':
            return pd.DataFrame(result)
        elif format == 'ndarray':
            return result
        else:
            raise ValueError(f"{format} is an invalid value for 'format'")

    def setCheckpoint(self, checkpoint, filename="parmoo"):
        """ Activate checkpointing.

        Args:
            checkpoint (bool): Turn checkpointing on (True) or off (False).
            filename (str, optional): Set the base checkpoint filename/path.
                The checkpoint file will have the JSON format and the
                extension ".simdb.json" appended to the end of filename.

        """

        if not isinstance(checkpoint, bool):
            raise TypeError("checkpoint must have the bool type")
        if not isinstance(filename, str):
            raise TypeError("filename must have the string type")
        self.checkpoint_data = checkpoint
        self.checkpoint_file = filename
        if self.running:
            self._checkpoint_metadata(self.checkpoint_file)
            # Save any pre-existing data
            for sim_name in self.sim_db:
                for i in range(self.sim_db[sim_name]['n']):
                    self.checkpointSimData(
                        self.sim_db[sim_name]['x_vals'][i],
                        self.sim_db[sim_name]['s_vals'][i],
                        sim_name,
                        self.checkpoint_file
                    )
            for i in range(self.obj_db['n']):
                self.checkpointObjData(
                    self.obj_db['x_vals'][i],
                    self.obj_db['f_vals'][i],
                    self.obj_db['c_vals'][i],
                    self.checkpoint_file
                )

    def checkpointSimData(self, x, sx, sim_name, filename="parmoo"):
        """ Append the given simulation data point to the checkpoint file.

        Args:
            x (dict or numpy structured element): The design value to append.
            sx (list or numpy array): The simulation output to append.
            sim_name (str): The simulation name/index to append to.
            filename (str, optional): The filepath to the checkpointing
                file(s). Do not include file extensions, they will be
                appended automatically. Defaults to the value "parmoo"
                (filename will be "parmoo.simdb.json").

        """

        # Unpack x/sx pair into a json-compatible dict for saving
        toadd = {
            'name': sim_name
        }
        for dname in self.des_schema:
            key = dname[0]
            dtype = dname[1]
            if dtype[0] in ["i", "u"]:
                toadd[key] = int(x[key])
            elif dtype[0] in ["f"]:
                toadd[key] = float(x[key])
            else:
                toadd[key] = str(x[key])
        toadd['out'] = [float(sxi) for sxi in sx]
        fname = f"{filename}.simdb.json"
        # Append new entries to a new line in existing file
        with open(fname, 'a') as fp:
            print(file=fp)
            json.dump(toadd, fp)

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

        # Unpack x/fx/cx into a json-compatible dict for saving
        toadd = {
            'name': "obj_db"
        }
        for dname in self.des_schema:
            key = dname[0]
            dtype = dname[1]
            if dtype[0] in ["i", "u"]:
                toadd[key] = int(x[key])
            elif dtype[0] in ["f"]:
                toadd[key] = float(x[key])
            else:
                toadd[key] = str(x[key])
        for oname in self.obj_schema:
            toadd[oname[0]] = float(fx[oname[0]])
        for cname in self.con_schema:
            toadd[cname[0]] = float(cx[cname[0]])
        fname = f"{filename}.simdb.json"
        # Append new entries to a new line in existing file
        with open(fname, 'a') as fp:
            print(file=fp)
            json.dump(toadd, fp)

    def loadCheckpoint(self, filename="parmoo"):
        """ Reload from the given checkpoint file.

        Args:
            filename (str, optional): The filepath to the checkpointing
                file(s). Do not include file extensions, they will be
                appended automatically. Defaults to the value "parmoo"
                (filename will be "parmoo.simdb.json").

        """

        if not self.isEmpty():
            raise RuntimeError(
                "Attempting to load a previous checkpoint but the database"
                " is non empty. Proceeding could overwrite existing"
                " data or create incosistent states. Please save any existing"
                " data and reset the database before proceeding."
            )
        with open(f"{filename}.simdb.json", 'r') as fp:
            for i, linei in enumerate(fp):
                entryi = json.loads(linei)
                if 'name' not in entryi:
                    raise IOError(
                        f"{filename}.simdb.json contains an invalid entry."
                    )
                elif i == 0:
                    if entryi['name'] != 'metadata':
                        raise IOError(
                            f"{filename}.simdb.json is missing the metadata"
                            " header."
                        )
                    self.des_schema = [
                        tuple(tj) for tj in entryi['des_schema']
                    ]
                    self.sim_schema = [
                        tuple(tj) for tj in entryi['sim_schema']
                    ]
                    self.obj_schema = [
                        tuple(tj) for tj in entryi['obj_schema']
                    ]
                    self.con_schema = [
                        tuple(tj) for tj in entryi['con_schema']
                    ]
                    self.des_tols = entryi['des_tols']
                    self.checkpoint_data = False  # Disable temporarily
                    self.checkpoint_file = filename
                    self.checkpoint_new = False
                    self.startDatabase()
                elif entryi['name'] == 'obj_db':
                    x = {}
                    for key in self.des_schema:
                        x[key[0]] = entryi[key[0]]
                    fx = {}
                    for key in self.obj_schema:
                        fx[key[0]] = entryi[key[0]]
                    cx = {}
                    for key in self.con_schema:
                        cx[key[0]] = entryi[key[0]]
                    self.updateObjDb(x, fx, cx)
                elif entryi['name'] == "get_new_data":
                    for key in self.sim_db:
                        self.sim_db[key]['n_old'] = self.sim_db[key]['n']
                else:
                    x = {}
                    for key in self.des_schema:
                        x[key[0]] = entryi[key[0]]
                    sx = entryi['out']
                    sim_name = entryi['name']
                    self.updateSimDb(x, sx, sim_name)
        self.checkpoint_data = True  # Re-enable

    def _checkpoint_metadata(self, filename="parmoo"):
        """ Private helper to write metadata to the checkpoint file. """

        fname = f"{filename}.simdb.json"
        # Don't overwrite existing data when the new data flag is set
        if self.checkpoint_new and file_exists(fname):
            raise OSError(
                f"Creating a new save file, but {filename}.simdb.json already"
                " exists! Move the existing file to a new location, delete it"
                " or load it first so that ParMOO doesn't overwrite your"
                " existing data..."
            )
        with open(fname, "w") as fp:
            json.dump({
                'name': "metadata",
                'des_schema': self.des_schema,
                'sim_schema': self.sim_schema,
                'obj_schema': self.obj_schema,
                'con_schema': self.con_schema,
                'des_tols': self.des_tols,
            }, fp)
        self.checkpoint_new = False

    def _log_new_data_call(self, filename="parmoo"):
        """ Private helper to log when new data was requested. """

        fname = f"{filename}.simdb.json"
        with open(fname, "a") as fp:
            print(file=fp)
            json.dump({'name': "get_new_data"}, fp)

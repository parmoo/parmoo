
""" Contains the SimulationDatabase class for storing simulation results.

``parmoo.SimulationDatabase`` is the base class for storing multiobjective
simulation outputs. Each SimulationDatabase object may contain several
simulations, and their corresponding objective and constraint violation scores.

"""

import inspect
from jax import numpy as jnp
import json
import logging
import numpy as np
from os.path import exists as file_exists
import pandas as pd

from parmoo.util import check_names, updatePF, approx_equal


class SimulationDatabase:
    """ A Database class for simulation optimization problems (SOPs)
    specialized for storing and filtering multiobjective data.

    To define the SimulationDatabase, add each design variable, simulation,
    objective, and constraint by using the following functions:
     * ``SimulationDatabase.addDesign(*args)``
     * ``SimulationDatabase.addSimulation(*args)``
     * ``SimulationDatabase.addObjective(*args)``
     * ``SimulationDatabase.addConstraint(*args)``

    After creating a SimulationDatabase, the following methods may be useful
    for getting the numpy.dtype of the input/output arrays:
     * ``SimulationDatabase.getDesignType()``
     * ``SimulationDatabase.getSimulationType()``
     * ``SimulationDatabase.getObjectiveType()``
     * ``SimulationDatabase.getConstraintType()``

    To add simulation data, use:
     * ``SimulationDatabase.checkSimDb(x, sim_name)``
     * ``SimulationDatabase.updateSimDb(x, sx, sim_name)``

    Once all simulation's have been updated for a given x, the objective
    database can be updated using:
     * ``SimulationDatabase.addObjData(x, fx, cx)``

    To force a save (checkpoint) of the current state of the simulation
    database, use:
     * ``MOOP.savedata(x, sx, sim_name, [filename="parmoo"])``

    Finally, the following methods are used to retrieve (filtered) simultation
    and objective data:
     * ``SimulationDatabase.getPF(format='ndarray')``
     * ``SimulationDatabase.getSimulationData(format='ndarray')``
     * ``SimulationDatabase.getObjectiveData(format='ndarray')``

    """

    __slots__ = [
        # Schemas
        'des_schema', 'sim_schema', 'obj_schema', 'con_schema',
        # Design tolerances for lookup
        'des_tols',
        # Compiled flag
        'compiled',
        # Checkpointing markers
        'checkpoint_data', 'checkpoint_file', 'new_data',
        # Database information
        'data', 'sim_db',
    ]

    def __init__(self):
        """ Initializer for the SimulationDatabase class. """

        # Initialize the schemas
        self.des_schema, self.sim_schema = [], []
        self.obj_schema, self.con_schema = [], []
        # Initialize design tolerances for lookup
        self.des_tols = {}
        # Initialize the compiled flag
        self.compiled = False
        # Initialize checkpointing markers
        self.checkpoint_data = False
        self.checkpoint_file = "parmoo"
        self.new_data = True
        # Initialize the database
        self.obj_db, self.sim_db = {}, {}

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

        check_names(
            name,
            self.des_schema, self.sim_schema, self.obj_schema, self.con_schema
        )
        self.des_schema.append((name, dtype))
        self.des_tols[name] = des_tol

    def addSimulation(self, name, m):
        """ Add new simulations to the SimulationDatabase schema.

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
        """ Add a new objective to the SimulationDatabase schema.

        Args:
            name (str): The unique name of this objective output.

        """

        check_names(
            name,
            self.des_schema, self.sim_schema, self.obj_schema, self.con_schema
        )
        self.obj_schema.append((name, 'f8'))

    def addConstraint(self, name):
        """ Add a new constraint to the SimulationDatabase schema.

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
        """ Initialize the SimulationDatabase. """

        # For safety reasons, don't let silly users delete their data
        if (
            self.obj_db['n'] > 0 or
            any([self.sim_db[key]['n'] > 0 for key in self.sim_db])
        ):
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
        self.compiled = True
        logging.info("   Done.")

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

            sx (ndarray): A 1D array containing the corresponding
                simulation output(s).

            sim_name (str): The name of the simulation to whose database the
                pair (x, sx) will be added into.

        """

        if not self.compiled:
            raise RuntimeError(
                "Cannot begin adding items to the database before compiling"
            )
        if sim_name not in self.sim_db:
            raise ValueError(f"{sim_name} is not a legal name/index")
        if (
            len(self.sim_db[sim_name]['x_vals']) !=
            len(self.sim_db[sim_name]['s_vals'])
        ):
            raise RuntimeError(f"{sim_name} database has become inconsistent")
        i = self.sim_db[sim_name]['n']
        # Check if database needs to be resized
        if i >= len(self.sim_db[sim_name]['x_vals']):
            self.sim_db[sim_name]['x_vals'] = np.append(
                self.sim_db[sim_name]['x_vals'],
                np.zeros(i, dtype=self.des_schema), axis=0
            )
            self.sim_db[sim_name]['s_vals'] = np.append(
                self.sim_db[sim_name]['s_vals'],
                np.zeros((i, sx.size)), axis=0
            )
        for key in x:
            self.sim_db[sim_name]['x_vals'][key][i] = x[key]
        self.sim_db[sim_name]['s_vals'][i, :] = sx
        self.sim_db[sim_name]['n'] += 1
        # If various checkpointing modes are on, then save the current states
        if self.checkpoint_data:
            self.savedata(x, sx, sim_name, filename=self.checkpoint_file)

    def updateObjDb(self, x, fx, cx):
        """ Update the internal objective database with a true evaluation of x.

        Args:
            x (dict): A Python dictionary containing the value of the design
                variable to add to ParMOO's database.
            fx (numpy.ndarray): An array of objective values to add.
            cx (numpy.ndarray): An array of constraint values to add.

        """

        if not self.compiled:
            raise RuntimeError(
                "Cannot begin adding items to the database before compiling"
            )
        if (
            len(self.obj_db['x_vals']) != len(self.obj_db['f_vals']) !=
            len(self.obj_db['c_vals'])
        ):
            raise RuntimeError(f"objective database has become inconsistent")
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
        for key in x:
            self.obj_db['x_vals'][key][i] = x[key]
        for key in fx:
            self.obj_db['f_vals'][key][i] = fx[key]
        for key in cx:
            self.obj_db['c_vals'][key][i] = cx[key]
        self.obj_db['n'] = 1

    def browseSimDb(self):
        """ Browse all design values that are present in every sim database.

        Yields:
            A sequence of tuples (x, sx) where each x is a (dict) design point
            that is present in every internal simulation database, and sx is a
            dictionary of simulation outputs from each of these database.

        """

        if len(self.sim_schema) > 0:
            sim0 = self.sim_schema[0]
            n0 = self.sim_db[sim0]['n']
            for xi, sxi in zip(
                self.sim_db[sim0[0]]['x_vals'][:n0],
                self.sim_db[sim0[0]]['s_vals'][:n0]
            ):
                # Initialize the x vals and s vals
                x_vals = {}
                for name in self.des_schema:
                    x_vals[name[0]] = xi[name]
                if len(sim0) > 2:
                    s_vals = {sim0[0]: sxi.copy()}
                else:
                    s_vals = {sim0[0]: sxi[0]}
                # Look for matches in all other sim_dbs
                matched = True
                for simi in self.sim_schema[1:]:
                    matched = False
                    ni = self.sim_db[simi[0]]['n']
                    for xj, sxj in zip(
                        self.sim_db[simi[0]]['x_vals'][:ni],
                        self.sim_db[simi[0]]['s_vals'][:ni]
                    ):
                        if approx_equal(x_vals, xj, des_tols):
                            if len(simi) > 2:
                                s_vals[simi[0]] = sxj.copy()
                            else:
                                s_vals[simi[0]] = sxj[0]
                            matched = True
                            break  # Break once we found a match in simi's db
                    if not matched:
                        break  # Break if there was no match in simi's db
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

        # Sort the objective values
        n = self.obj_db['n']
        lex_inds = np.lexsort([
            self.obj_db['f_vals'][obj[0]][:n] for obj in obj_schema
        ])
        # Loop over all points and look for nondominated points
        nondom_out = {
            'x_vals': np.zeros(n, dtype=self.des_schema),
            'f_vals': np.zeros(n, dtype=self.obj_schema),
            'c_vals': np.zeros(n, dtype=self.con_schema)
        }
        ndpts = 0
        for i in lex_inds:
            if (
                np.all(self.obj_db['c_vals'] < 1e-8) and np.all(np.any(
                    self.obj_db['f_vals'][i] <
                    nondom_out['f_vals'][:ndpts, :], axis=1
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
            result[dt[0]] = nondom_out['f_vals'][dt[0]][:ndpts]
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
                result[sname[0]]['out'] = self.sim_db[sname[0]]['s_vals'][:n, 0]
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
                    rtempi['out'] = result[snamei]['out'][:, 0]
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

        # Initialize result array
        n = self.obj_db['n']
        result = np.zeros(
            n, dtype=(self.des_schema + self.obj_schema + self.con_schema)
        )
        # Extract all results
        if self.obj_db['n'] > 0:
            for i, xi in enumerate(self.obj_db['x_vals']):
                for (name, t) in self.des_schema:
                    result[name][i] = xi[name]
            for i, (name, t) in enumerate(self.obj_schema):
                result[name][:] = self.obj_db['f_vals'][:, i]
            for i, (name, t) in enumerate(self.con_schema):
                result[name][:] = self.obj_db['c_vals'][:, i]
        if format == 'pandas':
            return pd.DataFrame(result)
        elif format == 'ndarray':
            return result
        else:
            raise ValueError(f"{format} is an invalid value for 'format'")

    def savedata(self, x, sx, sim_name, filename="parmoo"):
        """ Save the current simulation database for this MOOP.

        Args:
            filename (str, optional): The filepath to the checkpointing
                file(s). Do not include file extensions, they will be
                appended automatically. Defaults to the value "parmoo"
                (filename will be "parmoo.simdb.json").

        """

        # Check whether file exists first
        exists = file_exists(f"{filename}.simdb.json")
        if exists and self.new_data:
            raise OSError(
                f"Creating a new save file, but {filename}.simdb.json already"
                " exists! Move the existing file to a new location or delete"
                " it so that ParMOO doesn't overwrite your existing data..."
            )
        # Unpack x/sx pair into a dict for saving
        toadd = {'sim_id': sim_name}
        for dname in self.des_schema:
            key = dname[0]
            if (
                np.issubdtype(x[key], np.integer) or
                jnp.issubdtype(x[key], jnp.integer)
            ):
                toadd[key] = int(x[key])
            elif (
                np.issubdtype(x[key], np.floating) or
                jnp.issubdtype(x[key], jnp.floating)
            ):
                toadd[key] = float(x[key])
            else:
                toadd[key] = str(x[key])
        if isinstance(sx, np.ndarray) or isinstance(sx, jnp.ndarray):
            toadd['out'] = [float(sxi) for sxi in sx]
        else:
            toadd['out'] = float(sx)
        # Save in file with proper extension
        fname = f"{filename}.simdb.json"
        with open(fname, 'a') as fp:
            json.dump(toadd, fp)
        self.new_data = False

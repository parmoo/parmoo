""" Use the MOOP.iterate() point generator as a MDML client.

Create and start up a MDML client and optimize the producer using the PARMOO
generator as the consumer.

"""

from parmoo import MOOP
import mdml_client as mdml
import numpy as np
import logging
import datetime


def dummy_sim(x):
    return 0.0

    
class make_objective():
    """ Helper class for making objectives.

    Creates a ParMOO objective based on "goal" input.

    """

    __slots__ = ['goal', 'name', 'i']

    def __init__(self, goal, name, i):
        """ Constructor for make_objective class.

        Args:
            goal (str): Either "min" to minimize a simulation output, or
                "max" to maximize a simulation output.

            name (str): The name of the simulation output to minimize/maximize.

            i (int): The index of the simulation output to minimize or
                maximize.

        """

        # Check for bad input
        if goal.lower() != "min" and goal.lower() != "max":
            raise(ValueError("Error in msg['value']['Dependent_Par']: ",
                             "Goal must be `min` or `max`, not `",
                             str(goal), "`..."))
        # Initialize internal variables
        self.goal = goal.lower()
        self.name = name
        self.i = i
        return

    def __call__(self, x, sx, der=0):
        """ Minimize or maximize a single simulation output.

        Args:
            x (np.ndarray): 1d array containing design point input

            sx (np.ndarray): 1d array containing simulation input (sx[name][i])
                is minimized or maximized

        """

        # Goal 1: Maximize
        if self.goal == 'max':
            # Derivative wrt x is always 0
            if der == 1:
                return np.zeros(1, dtype=x.dtype)[0]
            # Derivative wrt sx is -e_i, since we are maximizing
            elif der == 2:
                result = np.zeros(1, dtype=sx.dtype)[0]
                result[self.name][self.i] = -1.0
                return result
            # No derivative, return -sx[name][i]
            else:
                return -sx[self.name][self.i]
        # Goal 2: Minimize
        elif self.goal == 'min':
            # Derivative wrt x is always 0
            if der == 1:
                return np.zeros(1, dtype=x.dtype)[0]
            # Derivative wrt sx is e_i
            elif der == 2:
                result = np.zeros(1, dtype=sx.dtype)[0]
                result[self.name][self.i] = 1.0
                return result
            # No derivative, return identity function for sx[name][i]
            else:
                return sx[self.name][self.i]
        # Failure during initialization, raise an appropriate error
        else:
            raise(ValueError("Error in msg['value']['Dependent_Par']: ",
                             "Goal must be `min` or `max`, not `",
                             str(goal), "`..."))


class MDML_MOOP(MOOP):

    def __init__(self, HOST_ID, opt_func=None, hyperparams=None,
                 topic=None, defn_topic=None, send_topic=None,
                 recv_topic=None, stop_signal=None, group_id=None):
        """ Initializer for the MDML interface to the MOOP class.

        Args:
            HOST_ID (str): The address of the MDML host server.

            topic (str): The topic for this experiment.
                Defaults to "mdml-parmoo-[today's date]".

            defn_topic (str): The topic for definition messages.
                Defaults to topic + "-definition".

            send_topic (str): The topic that ParMOO will use to send
                experiment/simulation suggestions. Defaults to topic + "-EXP".

            recv_topic (str): The topic that ParMOO will read experiment/
                simulation results from. Defaults to topic + "-DATA".

            stop_signal (str): The stop signal topic. Send any message to this
                topic to stop ParMOO from running. Defaults to topic + "-stop".

            group_id (str): The MDML group ID that ParMOO will consume from.
                Defaults to topic + "-parmoo-solver".

            opt_func (SurrogateOptimizer): A solver for the surrogate problems.
                Defaults to LBFGSB.

            hyperparams (dict, optional): A dictionary of hyperparameters for
                the opt_func, and any other procedures that will be used.
                Use the "overall_timeout" key to adjust the MDML timeout
                (given in seconds). Defaults to 7 days.

        Returns:
            MDML_MOOP: A new MDML_MOOP object with no design variables,
            objectives, or constraints.

        """

        # Create a MOOP using super() class
        if opt_func is None:
            from parmoo.optimizers import LBFGSB
            optimizer = LBFGSB
        else:
            optimizer = opt_func
        super().__init__(optimizer, hyperparams=hyperparams)
        # Define the host ID
        self.HOST_ID = str(HOST_ID)
        # Define the MDML topics
        if topic is None:
            self.topic = "mdml-parmoo-" + str(datetime.datetime.now())
        else:
            self.topic = topic
        if defn_topic is None:
            self.INIT_TOPIC = self.topic + "-definition"
        else:
            self.INIT_TOPIC = init_topic
        if send_topic is None:
            self.SEND_TOPIC = self.topic + "-EXP"
        else:
            self.SEND_TOPIC = send_topic
        if recv_topic is None:
            self.RECV_TOPIC = self.topic + "-DATA"
        else:
            self.RECV_TOPIC = recv_topic
        if stop_signal is None:
            self.STOP_SIGNAL = self.topic + "-stop"
        else:
            self.STOP_SIGNAL = stop_signal
        if group_id is None:
            self.GROUP_ID = self.topic + "-parmoo-solver"
        else:
            self.GROUP_ID = group_id
        # Set the MDML timeout counter
        if hyperparams is not None and "overall_timeout" in hyperparams.keys():
            self.overall_timeout = hp['overall_timeout']
        else:
            self.overall_timeout = 604800 # 7 days
        return
   
    def msg_to_data(self, msg):
        """ Convert a MDML message into ParMOO data.

        Args:
            msg (dict): A single MDML message, containing result data for
                a single design point x, and its corresponding simulation
                output sx.

        Returns:
            x (np.array): A design point input matching self.getDesignType().

            sx (np.array): The corresponding simulation output matching
            self.getSimulationType().

        """

        x = np.zeros(1, dtype=self.getDesignType())[0]
        sx = np.zeros(1, dtype=self.getSimulationType())[0]
        for (des_name, dt) in self.des_names:
            if dt == 'f8':
                x[des_name] = float(msg['value'][des_name])
            elif dt == 'i4':
                x[des_name] = int(msg['value'][des_name])
            else:
                raise ValueError("des_names contains illegal value")
        for i, (obj_name, dt) in enumerate(self.obj_names):
            sx["cfr_out"][i] = float(msg['value'][obj_name])
        return x, sx
   
    def data_to_msg(self, x, sim_id, RunID):
        """ Convert a ParMOO design point into a MDML message.

        Args:
            x (np.array): A design point input matching self.getDesignType().

            sim_id (str): The ID for the simulation to evaluate
                (unused for now).

            RunID (int): The ID counter for this particular run.

        Returns:
            msg (dict): A single MDML message, containing the value for
            each field in x.

        """

        # Initialize msg with time and Run_ID
        msg = {'time': str(datetime.datetime.now()),
               'Run_ID': RunID}
        for i, (name, dt) in enumerate(self.des_names):
            if dt == 'f8':
                msg[name] = float(x[name])
            elif dt == 'i4':
                msg[name] = int(x[name])
            else:
                raise ValueError("des_names contains illegal value")
        return msg

    def load_definition(self, definition, database=None):
        """ Load a MDML_MOOP from a MDML definition message.

        Args:
            definition (list, json-like): A json-like object containing a
                list of json-like/MDML definition messages (typically a
                list containing just a single message).

                Each definition message must contain a 'value' field with
                the following required sub-fields:
                 * Continuous_Par (dict): A dictionary of continuous
                   parameters. Each key is used as a design variable name by
                   ParMOO, and contains a sub-dictionary with the following
                   fields:
                    * Low: The lower bound on legal values for this parameter
                    * High: The upper bound on legal values for this parameter
                 * Categorical_Par (dict): Dictionary of categorical
                   parameters. Each key is used as a design variable name by
                   ParMOO, and contains a sub-dictionary with the following
                   fields:
                    * Levels: The number of levels for this categorical input
                 * Dependent_Par (dict): A dictionary of simulation/experiment
                   outputs. Each key is used as a simulation and objective name
                   by ParMOO, and contains a sub-dictionary with the following
                   fields:
                    * Goal: Either "Min" to minimize this output, or "Max"
                      to maximize this output.

                The following additional sub-fields are optional:
                 * search_budget (int): Number of experiments/simulations
                   to run in the initial design of experiments (defaults to 15)
                 * weights (list): A list of scalarizing weights describing
                   one solution of particular interest (defaults to equal
                   weighting of all objectives)
                 * batch_size (int): The preferred batch size for this
                   experiment (after the initial design is completed) --
                   (defaults to 3)

            database (list, json-like, optional): A list of simulation/
                experiment results data.

        Todo:
            WARNING! This load function currently only supports a subset
            of ParMOO's full functionality. For full functionality, use
            a regular initialization sequence.

        """

        from parmoo.optimizers import LBFGSB
        from parmoo.surrogates import GaussRBF
        from parmoo.searches import LatinHypercube
        from parmoo.acquisitions import RandomConstraint, FixedWeights

        # Initialize lists and variables for creating the new MDML_MOOP
        n = 0
        o = 0
        lb = []
        ub = []
        lvls = []
        des_names = []
        cont_names = []
        cat_names = []
        obj_names = []
        obj_funcs = []
        # Read the definition message
        for msg in definition:
            logging.info(f" Received config data: {msg['value']}")
            logging.info(" Loading config data...")
            # Read design variables into lists
            for name in msg['value']['Continuous_Par'].keys():
                logging.info(f"   Reading continuous des var {name}...")
                n += 1
                des_names.append(name)
                cont_names.append(name)
                lb.append(float(msg['value']['Continuous_Par'][name]['Low']))
                ub.append(float(msg['value']['Continuous_Par'][name]['High']))
                logging.info("   Done.")
            if 'Categorical_Par' in msg['value'].keys():
                for name in msg['value']['Categorical_Par'].keys():
                    logging.info("   Reading categorical design var " +
                                 name + "...")
                    n += 1
                    des_names.append(name)
                    cat_names.append(name)
                    lvls.append(msg['value']['Categorical_Par'][name]
                                   ['Levels'])
                    logging.info("   Done.")
            # Read the simulation outputs into lists
            for i, name in enumerate(msg['value']['Dependent_Par'].keys()):
                logging.info(f"   Reading sim/exp output {name}...")
                o += 1
                obj_names.append(name)
                # If 'Goal' is given, add to objectives as well
                if 'Goal' in msg['value']['Dependent_Par'][name].keys():
                    goal = msg['value']['Dependent_Par'][name]['Goal']
                    logging.info(f"   Reading objective {goal} {name}...")
                    obj_funcs.append(make_objective(goal, "cfr_out", i))
                logging.info("   Done.")
            # Read optional parameter settings
            logging.info("   Reading optional inputs...")
            search_budget = 15
            if 'search_budget' in msg['value'].keys():
                search_budget = msg['value']['search_budget']
                logging.info(f"     found: search budget = {search_budget}")
            weights = None
            if 'weights' in msg['value'].keys():
                weights = np.asarray(msg['value']['weights'])
                assert(weights.size == o)
                logging.info(f"     found: weights = {weights}")
            q = 3
            if 'batch_size' in msg['value'].keys():
                q = msg['value']['batch_size']
                assert(isinstance(q, int))
                assert(q > 0)
                logging.info(f"     found: batch size = {q}")
            logging.info("   Done.")
        logging.info(" Done.")
        # Create a new MDML_MOOP object using the super() constructor
        logging.info(" Creating MOOP...")
        hp = {}
        for name in self.hyperparams.keys():
            hp[name] = self.hyperparams[name]
        hp = {'search_budget': search_budget}
        super().__init__(self.optimizer, self.hyperparams)
        # Add design variables
        for l, u, name in zip(lb, ub, cont_names):
            self.addDesign({'name': name, 'lb': l, 'ub': u,
                            'des_tol': 0.0002})
        for lvl, name in zip(lvls, cat_names):
            self.addDesign({'name': name, 'des_type': "categorical",
                            'levels': lvl})
        # Create one simulation with `o` outputs
        sim_dicts = []
        sim_dicts.append({'name': 'cfr_out',
                          'm': o,
                          'sim_func': dummy_sim,
                          'hyperparams': {'search_budget': search_budget},
                          'search': LatinHypercube,
                          'surrogate': GaussRBF})
        self.addSimulation(*sim_dicts)
        # Add o objectives
        for obj_func, obj_name in zip(obj_funcs, obj_names):
            self.addObjective({'name': obj_name, 'obj_func': obj_func})
        # Add q-1 RandomConstraint acquisitions
        for i in range(q-1):
            self.addAcquisition({'acquisition': RandomConstraint})
        # The last constraint uses FixedWeights and can be customized
        if weights is None:
            self.addAcquisition({'acquisition': FixedWeights})
        else:
            self.addAcquisition({'acquisition': FixedWeights,
                                 'hyperparams': {'weights': weights}})
        logging.info(" Done.")
        # Reload from the database, if needed
        if isinstance(database, list):
            logging.info(" Reloading database...")
            for msg in database:
                logging.info(f"   Reloading data: {msg}")
                x, sx = self.msg_to_data(msg)
                self.update_sim_db(x, sx, "cfr_out")
                logging.info("   Done.")
            logging.info(" Done.")
        return
    
    def reload_results(self, msg_list):
        """ Reload a MDML_MOOP from a previous MDML message list.

        Args:
            msg_list (list, json-like): A list of dictionaries containing
                the history of MDML messages to replay.

        """

        # Create reloading variables
        database = []
        xpts = []
        sims = []
        count = 0
        new_count = 0
        # Read all previous msgs and find those that update ParMOO's state
        logging.info(f" Reloading previous messages...")
        for msg in msg_list:
            # Reload the problem definition
            if msg['topic'] == self.INIT_TOPIC:
                logging.info(f"   Reloading config data: {msg['value']}...")
                self.load_definition([msg], database=database)
                new_count = 0
                logging.info("   Done.")
            # Check if results were received
            if msg['topic'] == self.RECV_TOPIC:
                logging.info(f"   Reloading results: {msg['value']}...")
                # Check that the index is correct
                ind = int(msg['value']['Run_ID'])
                if ind != count:
                    raise(ValueError("Received an out-of-order RunID: " +
                                     str(ind)))
                    break
                # Add msg to the list
                database.append(msg)
                x, sx = self.msg_to_data(msg)
                xpts.append(x)
                sims.append(sx)
                # Increment the count
                count += 1
                new_count += 1
                logging.info("   Done.")
        logging.info(" Done.")
        # Once we have reloaded all the points, run through ParMOO
        # and simulate however many iterations would have elapsed.
        logging.info(f" Replaying ParMOO...")
        npts = count - new_count # point counter
        while npts < count:
            logging.info(f"   Iteration: {self.iteration}")
            batch = self.iterate(self.iteration)
            batch_size = len(batch)
            sim_batch = []
            for i in range(batch_size):
                self.update_sim_db(xpts[npts], sims[npts]["cfr_out"], "cfr_out")
                sim_batch.append((xpts[npts], "cfr_out"))
                npts += 1
                if npts == count:
                    break
            if npts <= count:
                self.updateAll(self.iteration, sim_batch)
                self.iteration += 1
        logging.info(" Done.")
        logging.info(f" {self.n_dat} results reloaded.")
        return
    
    def send_experiments(self, producer, batch, RunIDs):
        """ Send experiments via MDML producer.produce() method.

        Args:
            producer (MDML producer): The producer object.

            batch (list, ParMOO batch): A batch of ParMOO simulation points.

            RunIDs (list): The list of unique integer IDs for each experiment
                in batch.

        """
        # For each suggested experiment - clean up the data and produce
        for i, xi in enumerate(batch):
            d = self.data_to_msg(xi, "cfr_out", RunIDs[i])
            producer.produce(d)
            logging.info(f"   sent design: {d}")
        producer.flush()
        return
   
    # Receive function
    def receive_results(self, consumer, batch_size, RunIDs):
        """ Receive MDML experiments using the consumer.consume() method.

        Args:
            consumer (MDML consumer)

        """

        sim_out = np.zeros((batch_size, self.m_total))
        x_out = np.zeros(batch_size, dtype=self.getDesignType())
        count = 0
        for msg in consumer.consume(overall_timeout=self.overall_timeout):
            # Check if results were received
            if msg['topic'] == self.RECV_TOPIC:
                logging.info(f" received results: {msg['value']}")
                try:
                    ind = RunIDs.index(int(msg['value']['Run_ID']))
                except ValueError:
                    raise(ValueError("Received an invalid Run ID"))
                x, sx = self.msg_to_data(msg)
                x_out[ind] = x
                sim_out[ind] = sx["cfr_out"]
                # Check if we have gotten all of the results yet?
                count += 1
                if count == batch_size:
                    return sim_out, x_out, True
            # Check for stop signal
            elif msg['topic'] == self.STOP_SIGNAL:
                return None, None, False
        return None, None, False

    def solve(self, budget):
        """ Solve the MOOP using MDML with ParMOO.

        Args:
            budget (int): The number of iterations of ParMOO's solver to run.

        """
    
        # Generate example data for schema
        EX_SEND_DATA = {'Run_ID': 0, 'time': "hello world"}
        for (name, dt) in self.obj_names:
            EX_SEND_DATA[name] = 0.0
        for (name, dt) in self.des_names:
            if dt == 'f8':
                EX_SEND_DATA[name] = 0.0
            elif dt == 'i4':
                EX_SEND_DATA[name] = 0
            else:
                raise ValueError("des_names[" + str(i) + "] contains an " +
                                 "illegal value")
        # Generate the MDML schema
        schema = mdml.create_schema(EX_SEND_DATA,
                                    "PARMOO send schema",
                                    "schema sent from PARMOO to CFR")
        # Create MDML producer
        producer = mdml.kafka_mdml_producer(
            topic = self.SEND_TOPIC,
            schema = schema,
            kafka_host = self.HOST_ID,
            schema_host = self.HOST_ID)
        # Create MDML consumer
        consumer = mdml.kafka_mdml_consumer(
            topics = [self.RECV_TOPIC, self.STOP_SIGNAL],
            group = self.GROUP_ID,
            kafka_host = self.HOST_ID,
            schema_host = self.HOST_ID)
    
        # Randomize results? (comment below)
        np.random.seed(10252021)
   
        # Check the simulation counter
        counter = 0
        for db in self.sim_db:
            counter += db['n']
        # Perform iterations until budget is exceeded
        start = self.iteration
        for k in range(start, budget + 1):
            # Track the iteration counter
            self.iteration = k
            # Generate a batch by running one iteration
            sim_batch = self.iterate(k)
            # Sort the batch before sending
            batch_size = 0
            xbatch = []
            ibatch = []
            # Add good points to batch
            for (x, i) in sim_batch:
                xbatch.append(x)
                ibatch.append(i)
                batch_size += 1
            # Stable sort the batch by each key before sending with MDML
            # So the last entry in self.des_names is the primary sort key
            for (name, dt) in self.des_names:
                tbatch = []
                for x in xbatch:
                    tbatch.append(x[name])
                # Argsort the batch by contents of this name field, in order
                inds = np.asarray(np.argsort(np.asarray(tbatch),
                                             kind="stable"))
                xbatch = np.asarray(xbatch)[inds]
                ibatch = np.asarray(ibatch)[inds]
            # Send and receive each point in the batch
            status = False
            for i in range(batch_size):
                # Send and receive next item in the batch
                RunIDs = [counter]
                self.send_experiments(producer, [xbatch[i]], RunIDs)
                sim_out, x_out, status = self.receive_results(consumer, 1,
                                                              RunIDs)
                counter += 1
                # Check that a full batch was received
                if not status:
                    logging.info(" Experiments terminated prematurely")
                    break
                # Update the PARMOO databases
                self.update_sim_db(x_out[0], sim_out[0, :], "cfr_out")
            # Break the outer loop if status was False
            if not status:
                break
            # Update the PARMOO models
            self.updateAll(k, sim_batch)
        # Close the consumer object
        consumer.close()
        return

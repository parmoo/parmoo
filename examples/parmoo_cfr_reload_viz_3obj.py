""" Reload ParMOO experiment data from CFR/MDML log-files. """

INIT_FILE  = 'sample-config.json'
RELD_FILE  = 'reformatted_data_edit.json'
INIT_TOPIC  = 'mdml-CFR-experiment-01-definition'
SEND_TOPIC  = 'mdml-CFR-experiment-01-EXP'
RECV_TOPIC  = 'mdml-CFR-experiment-01-DATA'
STOP_SIGNAL = 'mdml-CFR-experiment-01-stop'
GROUP_ID    = 'mdml-CFR-experiment-01-tyler-g1'

RELOAD = True
RESTART = False

def parmoo_funcx(initializer=None, reloader=None):

    import time
    import numpy as np
    from parmoo import MOOP
    from parmoo.optimizers import LBFGSB
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.acquisitions import RandomConstraint, FixedWeights
    from parmoo import viz

    # Check whether we wil be reloading, restarting, or both
    RELOAD = True
    RESTART = True
    if initializer is None and reloader is None:
        raise ValueError("Must specify either initializer, reloader, or both")
    else:
        if initializer is None:
            RESTART = False
        if reloader is None:
            RELOAD = False

    # Helper function makes objectives
    def make_objective(goal, i):

        if goal.lower() == 'max':
            return lambda x, s, der=0: -s["cfr_out"][i] if der == 0 \
                                       else (np.zeros(1, x.dtype)[0]
                                             if der == 1 else
                                             np.array([(-np.eye(2)[i], )],
                                                       dtype=[('cfr_out',
                                                               float, 2)]))
        elif goal.lower() == 'min':
            return lambda x, s, der=0: s['cfr_out'][i] if der == 0 \
                                       else (np.zeros(1, x.dtype)[0]
                                             if der == 1 else
                                             np.array([(np.eye(2)[i], )],
                                                       dtype=[('cfr_out',
                                                               float, 2)]))
        else:
            raise(ValueError("Error in msg['value']['Dependent_Par']: ",
                             "Goal must be `min` or `max`, not `",
                             str(goal), "`..."))

    def min_RT(x, sx, der=0):
        if der==0:
            return x["RT"]
        elif der==1:
            ret_val = np.zeros(1, dtype=x.dtype)[0]
            ret_val["RT"] = 1.0
            return ret_val
        else:
            return np.zeros(1, dtype=sx.dtype)[0]

    # Init function
    def read_config(initializer, database=None):
        lb = []
        ub = []
        lvls = []
        n = 0
        o = 0
        des_names = []
        cont_names = []
        cat_names = []
        obj_names = []
        obj_funcs = []
        for msg in initializer:
            print(f"Received config data: {msg['value']}")
            for name in msg['value']['Continuous_Par'].keys():
                n += 1
                des_names.append(name)
                cont_names.append(name)
                lb.append(float(msg['value']['Continuous_Par'][name]['Low']))
                ub.append(float(msg['value']['Continuous_Par'][name]['High']))
            for name in msg['value']['Categorical_Par'].keys():
                n += 1
                des_names.append(name)
                cat_names.append(name)
                lvls.append(msg['value']['Categorical_Par'][name]['Levels'])
            for i, name in enumerate(msg['value']['Dependent_Par'].keys()):
                o += 1
                obj_names.append(name)
                goal = msg['value']['Dependent_Par'][name]['Goal']
                obj_funcs.append(make_objective(goal, i))
            # Read optional parameter settings
            search_budget = 15
            if 'search_budget' in msg['value'].keys():
                search_budget = msg['value']['search_budget']
            weights = None
            if 'weights' in msg['value'].keys():
                weights = np.asarray(msg['value']['weights'],
                                     dtype=np.float64)
                assert(weights.size == o)
        # Create one simulation with `o` outputs
        sim_dicts = []
        if database is None:
            database = {}
        sim_dicts.append({'name': 'cfr_out',
                          'm': o,
                          'sim_func': lambda x: 1.0,
                          'hyperparams': {'search_budget': search_budget},
                          'search': LatinHypercube,
                          'surrogate': GaussRBF,
                          'sim_db': database})
        # Create the MOOP object
        moop = MOOP(LBFGSB, hyperparams={})
        for l, u, name in zip(lb, ub, cont_names):
            moop.addDesign({'name': name, 'lb': l, 'ub': u, 'des_tol': 0.0002})
        for lvl, name in zip(lvls, cat_names):
            moop.addDesign({'name': name, 'des_type': "categorical",
                            'levels': lvl})
        moop.addSimulation(*sim_dicts)
        # Add 2 objectives
        moop.addObjective({'name': "-TFMC Int",
                           'obj_func': obj_funcs[0]})
        moop.addObjective({'name': "Byprod. Int",
                           'obj_func': obj_funcs[1]})
        moop.addObjective({'obj_func': min_RT, 'name': "Reaction Time"})
        # Add 2 RandomConstraint acquisitions
        for i in range(2):
            moop.addAcquisition({'acquisition': RandomConstraint})
        # The last acqusition can be customized
        if weights is None:
            moop.addAcquisition({'acquisition': FixedWeights})
        else:
            moop.addAcquisition({'acquisition': FixedWeights,
                                 'hyperparams': {'weights': weights}})
        return moop, des_names, obj_names

    # Receive function
    def reload_results(reloader):
        # Create reloading variables
        sims = []
        xpts = []
        count = 0
        new_count = 0
        # Read all previous msgs and find those that update ParMOO's state
        for msg in reloader:
            # Reload the problem definition
            if msg['topic'] == INIT_TOPIC:
                print(f"Reloaded config data: {msg['value']}")
                moop, des_names, obj_names = read_config([msg],
                                                         database={'x_vals':
                                                                   xpts,
                                                                   's_vals':
                                                                   sims})
                new_count = 0
            # Check if results were received
            if msg['topic'] == RECV_TOPIC:
                print(f"reloaded results: {msg['value']}")
                # Check that the index is correct
                ind = int(msg['value']['Run_ID'])
                if ind != count:
                    raise(ValueError("Received an out-of-order RunID: " +
                                     str(ind)))
                    break
                # Read xi and si and add to the list
                xi = np.zeros(1, dtype=np.dtype(moop.des_names))
                si = np.zeros(len(obj_names))
                for j, (name, dt) in enumerate(moop.des_names):
                    xi[name] = np.array(msg['value'][name]).astype(dt)
                for j, name in enumerate(obj_names):
                    si[j] = float(msg['value'][name])
                xpts.append(xi)
                sims.append(si)
                # Increment the count
                count += 1
                new_count += 1
        # Once we have reloaded all the points, run through PARMOO
        # and simulate however many iterations would have elapsed.
        k = 0 # iteration counter
        npts = count - new_count # point counter
        while npts < count:
            batch = moop.iterate(k)
            batch_size = len(batch)
            sim_batch = []
            for i in range(batch_size):
                moop.update_sim_db(xpts[npts], sims[npts], 0)
                sim_batch.append((xpts[npts], 0))
                npts += 1
                if npts == count:
                    break
            moop.updateAll(k, sim_batch)
            k += 1
        return moop, des_names, obj_names, k, npts

    # If reloading
    start = 0
    npts = 0
    if RELOAD:
        # Reload all the points into the MOOP
        moop, des_names, obj_names, start, npts = reload_results(reloader)
        # Do we want to restart ParMOO or resume?
        if RESTART:
            # Create a new MOOP when restarting
            db = {}
            db['s_vals'] = moop.sim_db[0]['s_vals']
            db['x_vals'] = np.asarray([moop.__extract__(xi)
                                       for xi in moop.sim_db[0]['x_vals']])
            moop, des_names, obj_names = read_config(initializer, database=db)
            # Reset the iteration counter
            start = 0

        # Get and print all the data after finishing the run
        soln = moop.sim_db[0]
        print("num pts: ", str(soln['x_vals'].shape[0]))
        #print("Data:")
        #for (xi, fi) in zip(soln['x_vals'], soln['f_vals']):
        #    print("x=[ ", end="")
        #    for xj in xi:
        #        print(xj, end=" ")
        #    print("],   f=[ ", end="")
        #    for fj in fi:
        #        print(abs(fj), end=" ")
        #    print("]")
        #print("")

    else:
        # Initialize ParMOO
        moop, des_names, obj_names = read_config(initializer)

#    # Add some constraints
#    def c1(x, s, der=0):
#        if der == 1:
#            out = np.zeros(1, x.dtype)
#            out["Solvents"] = 100.0
#            return out
#        elif der == 2:
#            return np.zeros(s.size)
#        else:
#            return 100.0 * x["Solvents"]
#
#    def c2(x, s, der=0):
#        if der == 1:
#            out = np.zeros(1, x.dtype)
#            if x["Bases"] > 1.0:
#                out["Bases"] = 100.0
#                return out
#            elif x[3] < 1.0:
#                out["Bases"] = -100.0
#                return out
#            else:
#                return np.zeros(x.size)
#        elif der == 2:
#            return np.zeros(s.size)
#        else:
#            return 100.0 * abs(x["Bases"] - 1.0)
#
#    moop.addConstraint({'constraint': c1})
#    moop.addConstraint({'constraint': c2})

    # Generate a batch by running one iteration
    sim_batch = moop.iterate(0)
    # Update the PARMOO models
    moop.updateAll(0, sim_batch)

    return moop

if __name__ == "__main__":
    import json
    from parmoo import viz

    reloader = open(RELD_FILE)
    data = json.load(reloader)
    reloader.close()

    moop = parmoo_funcx(initializer=None, reloader=data)

    print(moop.getPF())
    print(moop.getObjectiveType())
    print(moop.getDesignType())

    print('\nyour results are at http://127.0.0.1:8050/\n ')
    # print('if a [download dataset] button pops up,')
    # print('please ignore it, it's broken)

    # ! to change what's plotted, change which line of plotting code
    # ! is commented out and save the file.
    # ! you do not need to rerun the script
    # ! (we currently only support displaying one plot at a time)

    # viz.plot.scatter(moop, db='obj')
    viz.plot.scatter(moop, db='obj')
    # viz.plot.radar(moop, db='pf')
    # viz.plot.radar(moop, db='obj')
    # viz.plot.parallel_coordinates(moop, db='pf')
    # viz.plot.parallel_coordinates(moop, db='obj')

    # * db='pf' plots the pareto front
    # * db='obj' plots the objective data

    # * currently, ParMOO supports native plotting of
    # * scatterplot - viz.plot.scatter(moop)
    # * radar plot - viz.plot.radar(moop)
    # * parallel coordinates plot - viz.plot.parallel_coordinates(moop)

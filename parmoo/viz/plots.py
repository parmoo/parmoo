""" This module contains a library of common plotting functions.

The functions are:

 Print raw data to terminal:
  * ``printDataTypes(moop)`` -- print data type
  * ``printPF_raw(moop)`` -- print Pareto Front
  * ``printObjectiveData_raw(moop)`` -- print objective database
  * ``printSimulationData_raw(moop)`` -- print simulation database(s)
  * ``printDesignVariables_raw(moop)`` -- print design variable data
  * ``printConstraints_raw(moop)`` -- print constraint data
  * ``printMOOP_raw(moop)`` -- print all MOOP data

 Print data to terminal as table
  * ``printPF(moop)`` -- print Pareto Front
  * ``printObjectiveData(moop)`` -- print objective database
  * ``printSimulationData(moop)`` -- print simulation database(s)
  * ``printMOOP(moop)`` -- print all MOOP data

 Display data in interactive browser plot
  * ``scatter(moop)`` -- Plot MOOP results as matrix of 2D scatterplots
  * ``scatter3d(moop)`` -- Plot MOOP results as matrix of 3D scatterplots
  * ``radar(moop)`` -- Plot MOOP results as radar
  * ``parallel_coordinates(moop)`` -- Plot MOOP results as parallel coordinates
  * ``heatmap(moop)`` -- Plot MOOP results as heatmap
  * ``petal(moop)`` -- Plot MOOP results as petal diagram
  * ``radviz(moop)`` -- Plot MOOP results as RadViz
  * ``star_coordinates(moop)`` -- Plot MOOP results as star coordinates

 Development
  * ``dummyFunction(moop)`` -- place functions here for testing

"""

import plotly.express as px         # 15.2 MB package
# import plotly.graph_objects as go
from tabulate import tabulate       # 29 kB package
# from parmoo import MOOP
# import numpy as np

# des_type = moop.getDesignType()
# obj_type = moop.getObjectiveType()
# sim_type = moop.getSimulationType()
# const_type = moop.getConstraintType()
# pf = moop.getPF()
# obj_db = moop.getObjectiveData()
# sim_db = moop.getSimulationData()


def printDataTypes(moop):
    """ Print data types to terminal.

    Args:
        moop (MOOP): A ParMOO MOOP containing the data types to print.

    Returns:
        None

    """
    des_type = moop.getDesignType()
    obj_type = moop.getObjectiveType()
    sim_type = moop.getSimulationType()
    const_type = moop.getConstraintType()
    print("\nDATA TYPES FOR YOUR MOOP:\n")
    print("Design variable type:   " + str(des_type))
    print("Simulation output type: " + str(sim_type))
    print("Objective type:         " + str(obj_type))
    print("Constraint type:        " + str(const_type))

#
# ! THESE FUNCTIONS PRINT DATA TO THE TERMINAL IN BLOCK FORM
#


def printPF_raw(moop):
    """ Print Pareto Front to terminal as raw data.

    Args:
        moop (MOOP): A ParMOO MOOP containing the Pareto Front to print.

    Returns:
        None

    """
    print("\nPARETO FRONT:\n")
    pf = moop.getPF()
    print(pf)


def printObjectiveData_raw(moop):
    """ Print objective database to terminal as raw data.

    Args:
        moop (MOOP): A ParMOO MOOP containing the objective database to print.

    Returns:
        None

    """
    print("\nOBJECTIVE DATA:\n")
    obj_type = moop.getObjectiveType()
    obj_db = moop.getObjectiveData()
    for obj_key in obj_type.names:
        print(f"Objective: {obj_key}:")
        print(obj_db[obj_key])


def printSimulationData_raw(moop):
    """ Print simulation database(s) to terminal as raw data.

    Args:
        moop (MOOP): A ParMOO MOOP containing simulation database(s) to print.

    Returns:
        None

    """
    print("\nSIMULATION DATA:\n")
    sim_db = moop.getSimulationData()
    sim_type = moop.getSimulationType()
    for sim_key in sim_type.names:
        print(f"Simulation: {sim_key}:")
        print(sim_db[sim_key])


def printDesignVariables_raw(moop):
    """ Print design variable data to terminal as raw data.

    Args:
        moop (MOOP): A ParMOO MOOP containing design variable data to print.

    Returns:
        None

    """
    print("\nDESIGN VARIABLES:\n")
    des_type = moop.getDesignType()
    obj_db = moop.getObjectiveData()
    for des_key in des_type.names:
        print(f"Design Variable: {des_key}:")
        print(obj_db[des_key])


def printConstraints_raw(moop):
    """ Print constraint data to terminal as raw data.

    Args:
        moop (MOOP): A ParMOO MOOP containing the constraint data to print.

    Returns:
        None

    """
    print("\nCONSTRAINT DATA:\n")
    const_type = moop.getConstraintType()
    obj_db = moop.getObjectiveData()
    for const_key in const_type.names:
        print(f"Constraint: {const_key}:")
        print(obj_db[const_key])


def printMOOP_raw(moop):
    """ Print all MOOP data to terminal as raw data.

    Args:
        moop (MOOP): A ParMOO MOOP containing the data to print.

    Returns:
        None

    """
    printDataTypes(moop)
    printPF_raw(moop)
    printObjectiveData_raw(moop)
    printDesignVariables_raw(moop)
    # printConstraints_raw(moop)
    printSimulationData_raw(moop)
    print(" ")


#
# ! THESE FUNCTIONS PRINT DATA TO THE TERMINAL IN TABLE FORM
#


def printPF(moop):
    """ Print Pareto Front to terminal as table.

    Args:
        moop (MOOP): A ParMOO MOOP containing the Pareto Front to print.

    Returns:
        None

    """
    print("\nPARETO FRONT:\n")
    pf = moop.getPF()
    print(tabulate(pf, headers="keys"))
    # Pareto Front indices are reshuffled from the original order
    # in the .getPF() function
    # there is no information yielded by the indice locations themselves


def printObjectiveData(moop):
    """ Print objective database to terminal as table.

    Args:
        moop (MOOP): A ParMOO MOOP containing the objective database to print.

    Returns:
        None

    """
    print("\nOBJECTIVE DATABASE:\n")
    obj_db = moop.getObjectiveData()
    print(tabulate(obj_db, headers="keys", showindex=True))


def printSimulationData(moop):
    """ Print simulation database(s) to terminal as table(s).

    Args:
        moop (MOOP): A ParMOO MOOP containingW simulation database(s) to print.

    Returns:
        None

    """
    print("\nSIMULATION DATABASE:\n")
    sim_db = moop.getSimulationData()
    sim_type = moop.getSimulationType()
    for sim_key in sim_type.names:
        print(f"\nSimulation: {sim_key}:\n")
        print(tabulate(sim_db[sim_key], headers="keys", showindex=True))

        # sim_data = sim_db[sim_key]
        # for i in range(len(sim_data)):
        #     # print(str(i) + " ", end='')
        #     sim_line = sim_data[i]
        #     sim_out = sim_line[len(sim_line) - 1]
        #     for j in range(len(sim_line)):
        #         sim_column = sim_line[j]
        #         print(str(i) + " " + str(sim_column))

        # print simulation name
        # for each simulation, print header
        # after header, for each line print index
        # then data from each column
        # then data from the out column
        # then begin a new line
        # this all would eventually be nice to add,
        # but it's not a priority
        # for now, printing the sim_db without out-subkey headers is fine


def printMOOP(moop):
    """ Print all MOOP data to terminal in tables.

    Args:
        moop (MOOP): A ParMOO MOOP containing the data to print.

    Returns:
        None

    """
    printDataTypes(moop)
    printPF(moop)
    printObjectiveData(moop)
    printSimulationData(moop)

#
# ! THESE FUNCTIONS DISPLAY DATA IN interactive BROWSER PLOT
#


def scatter(moop):
    """ Display MOOP results as matrix of 2D scatterplots.

    Create an interactive plot that displays in the browser.

    For ``n`` objectives, generate an ``n x n`` matrix of 2D scatterplots.

    Users can hover above an output point to see input information pop up.

    Args:
        moop (MOOP): A ParMOO MOOP containing the MOOP results to plot.

    Returns:
        None

    """
    # des_type = moop.getDesignType()
    obj_type = moop.getObjectiveType()
    # sim_type = moop.getSimulationType()
    # const_type = moop.getConstraintType()
    pf = moop.getPF()
    obj_db = moop.getObjectiveData()
    # sim_db = moop.getSimulationData()

    # choose axes
    axes = []  # each axis relates to an objective
    for obj_key in obj_type.names:
        axes.append(obj_key)

    # sim_axes = [] # axes from the simulation database
    # for

    # plot pareto front
    pf_fig = px.scatter_matrix(pf,
                               dimensions=axes,
                               title="Pareto Front",
                               )
    pf_fig.update_traces(diagonal_visible=False)
    pf_fig.show()

    # plot objectives
    obj_fig = px.scatter_matrix(obj_db,
                                dimensions=axes,
                                title="Objective Data",
                                )
    obj_fig.update_traces(diagonal_visible=False)
    obj_fig.show()

    # plot simulation data
    # sim_fig = px.scatter_matrix(sim_db,
    #                             dimensions=axes,
    #                             title="Simulation Data",
    #                             )
    # sim_fig.update_traces(diagonal_visible=False)
    # sim_fig.show()


def scatter3d(moop):
    """ Display MOOP results as matrix of 3D scatterplots.

    Create an interactive plot that displays in the browser.

    For ``n`` objectives, generate an ``n x n`` matrix of 3D scatterplots.

    Args:
        moop (MOOP): A ParMOO MOOP containing the results to plot.

    Returns:
        None

    """
    pass


def radar(moop):
    """ Display MOOP results as radar plot.

    Create an interactive plot that displays in the browser.

    Args:
        moop (MOOP): A ParMOO MOOP containing the results to plot.

    Returns:
        None

    """

    # algo
    # each index in db gets a plot
    # for each objective db
        # objective[index] is the r for that plot
        

    # des_type = moop.getDesignType()
    obj_type = moop.getObjectiveType()
    # sim_type = moop.getSimulationType()
    # const_type = moop.getConstraintType()
    pf = moop.getPF()
    # obj_db = moop.getObjectiveData()
    # sim_db = moop.getSimulationData()

    # choose axes
    axes = []  # each axis relates to an objective
    for obj_key in obj_type.names:
        axes.append(obj_key)

    # plot pareto front
    pf_fig = px.line_polar(pf,
                           theta=axes,
                           r=pf[axes]
                           # title="Pareto Front",
                           )
    # pf_fig.update_traces(diagonal_visible=False)
    pf_fig.show()

    # # plot objectives
    # obj_fig = px.line_polar(obj_db,
    #                         theta=axes,
    #                         r=obj_db[axes]
    #                         # title="Objective Data",
    #                         )
    # # obj_fig.update_traces(diagonal_visible=False)
    # obj_fig.show()


def parallel_coordinates(moop):
    """ Display MOOP results as parallel coordinates plot.

    Create an interactive plot that displays in the browser.

    Users can select item(s) in a parallel coordinates plot
    by selecting an axis section which item(s) pass through.

    Args:
        moop (MOOP): A ParMOO MOOP containing the results to plot.

    Returns:
        None

    """
    obj_type = moop.getObjectiveType()
    pf = moop.getPF()
    obj_db = moop.getObjectiveData()

    # choose axes
    axes = []  # each axis relates to an objective
    for obj_key in obj_type.names:
        axes.append(obj_key)

    # plot pareto front
    pf_fig = px.parallel_coordinates(pf,
                                     labels=axes,
                                     title="Pareto Front",
                                     )
    pf_fig.show()

    # plot objectives
    obj_fig = px.parallel_coordinates(obj_db,
                                      labels=axes,
                                      title="Objective Data",
                                      )
    obj_fig.show()


def heatmap(moop):
    """ Display MOOP results as heatmap.

    Create an interactive plot that displays in the browser.

    Args:
        moop (MOOP): A ParMOO MOOP containing the results to plot.

    Returns:
        None

    """
    pass


def petal(moop):
    """ Display MOOP results as petal diagram.

    Create an interactive plot that displays in the browser.

    Args:
        moop (MOOP): A ParMOO MOOP containing the results to plot.

    Returns:
        None

    """
    pass


def radviz(moop):
    """ Display MOOP results as RadViz plot.

    Create an interactive plot that displays in the browser.

    Args:
        moop (MOOP): A ParMOO MOOP containing the results to plot.

    Returns:
        None

    """
    pass


def star_coordinates(moop):
    """ Display MOOP results as star coordinates plot.

    Create an interactive plot that displays in the browser.

    Args:
        moop (MOOP): A ParMOO MOOP containing the results to plot.

    Returns:
        None

    """
    pass

#
# ! TEST FUNCTION FOR DEVELOPMENT PURPOSES
# ! ALL EXAMPLES RUN DUMMY FUNCTION
#


def dummyFunction(moop):
    """ Dummy function for development purposes

    Functions to be tested in examples should be placed here.

    Args:
        moop (MOOP): A ParMOO MOOP for testing function(s) on.

    Returns:
        None
        (theoretically depends on test contents but
        currently all viz functions return None)

    """
    printMOOP_raw(moop)
    printMOOP(moop)
    parallel_coordinates(moop)
    scatter(moop)

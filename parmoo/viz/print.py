""" This module contains a library of common printing functions.

The functions are:

 Print data to terminal:
  * ``printDataTypes(moop)`` -- print data type
  * ``printPF(moop, style)`` -- print Pareto Front
  * ``printObjectiveData(moop, style)`` -- print objective database
  * ``printSimulationData(moop, style)`` -- print simulation database(s)
  * ``printMOOP(moop, style)`` -- print all MOOP data

"""

from tabulate import tabulate       # 29 kB package
# from parmoo import MOOP
# import numpy as np
# import warnings                     # native python package

# des_type = moop.getDesignType()
# obj_type = moop.getObjectiveType()
# sim_type = moop.getSimulationType()
# const_type = moop.getConstraintType()
# pf = moop.getPF()
# obj_db = moop.getObjectiveData()
# sim_db = moop.getSimulationData()


#
# ! THESE FUNCTIONS PRINT DATA TO THE TERMINAL
#


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


def printPF(moop, style='table'):
    """ Print Pareto Front to terminal as table.

    Args:
        moop (MOOP): A ParMOO MOOP containing the Pareto Front to print.
        style (String): Determines what formatting to apply to printed data.
            'table' - print data to terminal as table (default value)
            'raw' - print data to terminal without formatting

    Returns:
        None

    """

    # Pareto Front indices are reshuffled from the simulation order
    # there is no information yielded by the indice locations

    print("\nPARETO FRONT:\n")
    pf = moop.getPF()

    if (style == 'table'):
        print(tabulate(pf, headers="keys"))
    elif (style == 'raw'):
        print(pf)
    else:
        message = "'" + str(style) + "' not an acceptible value for 'style'\n"
        message += "Consider using 'table' or 'raw' instead."
        raise ValueError(message)


def printObjectiveData(moop, style='table'):
    """ Print objective database to terminal as table.

    Args:
        moop (MOOP): A ParMOO MOOP containing the objective database to print.
        style (String): Determines what formatting to apply to printed data.
            'table' - print data to terminal as table (default value)
            'raw' - print data to terminal without formatting

    Returns:
        None

    """

    print("\nOBJECTIVE DATA:\n")
    obj_db = moop.getObjectiveData()
    obj_type = moop.getObjectiveType()

    if (style == 'table'):
        print(tabulate(obj_db, headers="keys", showindex=True))
    elif (style == 'raw'):
        for obj_key in obj_type.names:
            print(f"Objective: {obj_key}:")
            print(obj_db[obj_key])
    else:
        message = "'" + str(style) + "' not an acceptible value for 'style'\n"
        message += "Consider using 'table' or 'raw' instead."
        raise ValueError(message)


def printSimulationData(moop, style='table'):
    """ Print simulation database(s) to terminal as table(s).

    Args:
        moop (MOOP): A ParMOO MOOP containingW simulation database(s) to print.
        style (String): Determines what formatting to apply to printed data.
            'table' - print data to terminal as table (default value)
            'raw' - print data to terminal without formatting

    Returns:
        None

    """

    print("\nSIMULATION DATA:\n")
    sim_db = moop.getSimulationData()
    sim_type = moop.getSimulationType()

    if (style == 'table'):
        for sim_key in sim_type.names:
            print(f"\nSimulation: {sim_key}:\n")
            print(tabulate(sim_db[sim_key], headers="keys", showindex=True))
    elif (style == 'raw'):
        for sim_key in sim_type.names:
            print(f"Simulation: {sim_key}:")
            print(sim_db[sim_key])
    else:
        message = "'" + str(style) + "' not an acceptible value for 'style'\n"
        message += "Consider using 'table' or 'raw' instead."
        raise ValueError(message)


def printMOOP(moop, style='table'):
    """ Print all MOOP data to terminal in tables.

    Args:
        moop (MOOP): A ParMOO MOOP containing the data to print.
        style (String): Determines what formatting to apply to printed data.
            'table' - print data to terminal as table (default value)
            'raw' - print data to terminal without formatting

    Returns:
        None

    """

    if (style == 'table') or (style == 'raw'):
        printDataTypes(moop)
        printPF(moop, style=style)
        printObjectiveData(moop, style=style)
        printSimulationData(moop, style=style)
    else:
        message = "'" + str(style) + "' not an acceptible value for 'style'\n"
        message += "Consider using 'table' or 'raw' instead."
        raise ValueError(message)

# def printDesignVariables_raw(moop):
#     """ Print design variable data to terminal as raw data.

#     Args:
#         moop (MOOP): A ParMOO MOOP containing design variable data to print.

#     Returns:
#         None

#     """
#     print("\nDESIGN VARIABLES:\n")
#     des_type = moop.getDesignType()
#     obj_db = moop.getObjectiveData()
#     for des_key in des_type.names:
#         print(f"Design Variable: {des_key}:")
#         print(obj_db[des_key])


# def printConstraints_raw(moop):
#     """ Print constraint data to terminal as raw data.

#     Args:
#         moop (MOOP): A ParMOO MOOP containing the constraint data to print.

#     Returns:
#         None

#     """
#     print("\nCONSTRAINT DATA:\n")
#     const_type = moop.getConstraintType()
#     obj_db = moop.getObjectiveData()
#     for const_key in const_type.names:
#         print(f"Constraint: {const_key}:")
#         print(obj_db[const_key])

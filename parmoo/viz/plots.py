import plotly.express as px
import plotly.graph_objects as go
from parmoo import MOOP

# des_type = moop.getDesignType()
# obj_type = moop.getObjectiveType()
# sim_type = moop.getSimulationType()
# const_type = moop.getConstraintType()
# pf = moop.getPF()
# obj_db = moop.getObjectiveData()
# sim_db = moop.getSimulationData()

# print Data Types to terminal
def printDataTypes(moop):
    des_type = moop.getDesignType()
    obj_type = moop.getObjectiveType()
    sim_type = moop.getSimulationType()
    const_type = moop.getConstraintType()
    print("\nDATA TYPES FOR YOUR MOOP:\n")
    print("Design variable type:   " + str(des_type))
    print("Simulation output type: " + str(sim_type))
    print("Objective type:         " + str(obj_type))
    print("Constraint type:        " + str(const_type))

# print Pareto Front to terminal
def printPF(moop):
    print("\nPARETO FRONT:\n")
    pf = moop.getPF()
    print(pf)

# print Objective Data to terminal
def printObjectiveData(moop):
    print("\nOBJECTIVE DATA:\n")
    obj_type = moop.getObjectiveType()
    obj_db = moop.getObjectiveData()
    for obj_key in obj_type.names:
        print(f"Objective: {obj_key}:")
        print(obj_db[obj_key])


def printDesignVariables(moop):
    print("\nDESIGN VARIABLES:\n")
    des_type = moop.getDesignType()
    obj_db = moop.getObjectiveData()
    for des_key in des_type.names:
        print(f"Design Variable: {des_key}:")
        print(obj_db[des_key])

def printConstraintData(moop):
    print("\nCONSTRAINT DATA:\n")
    const_type = moop.getConstraintType()
    obj_db = moop.getObjectiveData()
    for const_key in const_type.names:
        print(f"Constraint: {const_key}:")
        print(obj_db[const_key])

# get design variable from obj_db
# get constraint info ditto

# print by index


# # print Constraint Data to terminal
# def printConstraintData(moop):
#     print(" ")
#     print("Constraint Data:")
#     print(" ")

# print Simulation Data to terminal
def printSimulationData(moop):
    print("\nSIMULATION DATA:\n")
    sim_db = moop.getSimulationData()
    sim_type = moop.getSimulationType()
    for sim_key in sim_type.names:
        print(f"Simulation: {sim_key}:")
        print(sim_db[sim_key])

# print all MOOP data to terminal
def printMOOP(moop):
    printDataTypes(moop)
    printPF(moop)
    printObjectiveData(moop)
    printDesignVariables(moop)
    printConstraintData(moop)
    printSimulationData(moop)
    print(" ")

# print Objective Database as table
def printObjectiveTable(moop):
    for

# prints scatterplot in 2d
def scatter(moop):
    pass

# prints scatterplot in 3d
def scatter3d(moop):
    pass

# prints radar plot
def radar(moop):
    pass

# prints parallel coordinates plot
def parallel_coordinates(moop):
    pass

# prints heatmap
def heatmap(moop):
    pass

# prints petal diagram
def petal(moop):
    pass

# prints RadViz plot
def radviz(moop):
    pass

# prints star coordinates plot
def star_coordinates(moop):
    pass

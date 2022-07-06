""" This module contains a library of common plotting functions.

The functions are:

 Display data in interactive browser plot
  * ``scatter(moop)`` -- Plot MOOP results as matrix of 2D scatterplots
  * ``scatter3d(moop)`` -- Plot MOOP results as matrix of 3D scatterplots
  * ``radar(moop)`` -- Plot MOOP results as radar
  * ``parallel_coordinates(moop)`` -- Plot MOOP results as parallel coordinates
  * ``heatmap(moop)`` -- Plot MOOP results as heatmap
  * ``petal(moop)`` -- Plot MOOP results as petal diagram
  * ``radviz(moop)`` -- Plot MOOP results as RadViz
  * ``star_coordinates(moop)`` -- Plot MOOP results as star coordinates

 Utilities

  * ``dummyFunction(moop)`` -- place functions here for testing

For all interactve browser plot functions, there is a known issue
causing Plotly images to not export from Safari correctly. If you
encounter this issue, change your default browser to Chrome, Firefox,
or Edge.

"""

import plotly.express as px         # 15.2 MB package
import plotly.graph_objects as go
# import plotly.io as pio
# from parmoo import MOOP
# import numpy as np
import warnings                     # native python package

# des_type = moop.getDesignType()
# obj_type = moop.getObjectiveType()
# sim_type = moop.getSimulationType()
# const_type = moop.getConstraintType()
# pf = moop.getPF()
# obj_db = moop.getObjectiveData()
# sim_db = moop.getSimulationData()


#
# ! THESE FUNCTIONS DISPLAY DATA IN AN INTERACTIVE BROWSER PLOT
#


def scatter(moop, db='pf'):
    """ Display MOOP results as matrix of 2D scatterplots.

    Create an interactive plot that displays in the browser.

    For ``n`` objectives, generate an ``n x n`` matrix of 2D scatterplots.

    Users can hover above an output point to see input information pop up.

    Args:
        moop (MOOP): A ParMOO MOOP containing the MOOP results to plot.
        db (String): Indicates which database to plot.
                     Defaults to "pf".
                     Other options: "obj".
                     A keyword other than "pf" or "obj" throws an error.

    Returns:
        None

    """

    # * get info
    obj_type = moop.getObjectiveType()
    obj_db = moop.getObjectiveData()
    pf = moop.getPF()

    # * choose axes
    axes = []  # each axis relates to an objective
    for obj_key in obj_type.names:
        axes.append(obj_key)

    # * choose database
    if (db == 'pf'):
        database = pf
        plotTitle = "Pareto Front"
    elif db == 'obj':
        database = obj_db
        plotTitle = "Objective Data"
    else:
        message = "'" + str(db) + "' is not an acceptible value for 'db'\n"
        message += "Consider using 'pf' or 'obj' instead."
        raise ValueError(message)

    hoverInfo = []
    for key in database.dtype.names:
        redundant = False
        for ax in axes:
            if ax == key:
                redundant = True
        if redundant is False:
            hoverInfo.append(key)

    # * create plot
    fig = px.scatter_matrix(database,
                            dimensions=axes,
                            # color=,
                            title=plotTitle,
                            hover_data=hoverInfo
                            )
    fig.update_traces(diagonal_visible=False)



    # * display plot
    fig.show()


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


def radar(moop, db='pf'):
    """ Display MOOP results as radar plot.

    Create an interactive plot that displays in the browser.

    Args:
        moop (MOOP): A ParMOO MOOP containing the results to plot.
        db (String): Indicates which database to plot.
                     Defaults to "pf".
                     Other options: "obj".
                     A keyword other than "pf" or "obj" throws an error.

    Returns:
        None

    """
    # * get info
    obj_type = moop.getObjectiveType()
    obj_db = moop.getObjectiveData()
    pf = moop.getPF()

    # * setup axes
    axes = []
    wrap_around_key = ""
    wrap_around_count = 0
    for obj_key in obj_type.names:
        axes.append(obj_key)
        if wrap_around_count == 0:
            wrap_around_key = obj_key
        wrap_around_count += 1
    axes.append(wrap_around_key)

    # * choose database
    if (db == 'pf'):
        database = pf
        plotTitle = "Pareto Front"
    elif db == 'obj':
        database = obj_db
        plotTitle = "Objective Data"
    else:
        message = "'" + str(db) + "' is not an acceptible value for 'db'\n"
        message += "Consider using 'pf' or 'obj' instead."
        raise ValueError(message)

    # * raise warnings
    if len(axes) < 3:
        message = """
        Radar plots are best suited for MOOPs with at least 3 objectives.
        Consider using a scatterplot or parallel coordinates plot instead.
        """
        warnings.warn(message)

    # * create plot
    fig = go.Figure()
    for i in range(len(database)):
        traceName = (i)
        if plotTitle == "Pareto Front":
            hoverInfo = ""
        else:
            hoverInfo = "Design #" + str(i) + "\n"
        for key in database.dtype.names:
            hoverInfo += str(key)
            hoverInfo += ": "
            hoverInfo += str(database[key][i])
            hoverInfo += "<br>"
            # since plotly is JavaScript-based
            # it uses HTML string formatting
        values = []
        count = 0
        wrap_around_value = ""
        for obj_key in obj_type.names:
            values.append(database[obj_key][i])
            if count == 0:
                wrap_around_value = obj_key
            count += 1
        values.append(database[wrap_around_value][i])
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=axes,
            name=traceName,
            hovertext=hoverInfo
        ))

    # * improve aesthetics
    fig.update_traces(
        hoverinfo='text',
        selector=dict(
            type='scatterpolar'
        )
    )
    # fig.update_traces(
    #     marker_color=traceName,
    #     marker_coloraxis=traceName,
    #     selector=dict(
    #         type='scatterpolar'
    #     )
    # )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            )
        )
    )
    # combining the above and below functions will cause a SyntaxError
    fig.update_layout(
        title=dict(
            text=plotTitle
        )
    )
    if plotTitle == "Pareto Front":
        fig.update_layout(showlegend=True)
    else:
        fig.update_layout(showlegend=True)

    # * display plot
    fig.show()


def parallel_coordinates(moop, db='pf'):
    """ Display MOOP results as parallel coordinates plot.

    Create an interactive plot that displays in the browser.

    Users can select item(s) in a parallel coordinates plot
    by selecting an axis section which item(s) pass through.

    Args:
        moop (MOOP): A ParMOO MOOP containing the results to plot.
        db (String): Indicates which database to plot.
                     Defaults to "pf".
                     Other options: "obj".
                     A keyword other than "pf" or "obj" throws an error.

    Returns:
        None

    """
    # * get info
    obj_type = moop.getObjectiveType()
    pf = moop.getPF()
    obj_db = moop.getObjectiveData()

    # * setup axes
    axes = []  # each axis relates to an objective
    for obj_key in obj_type.names:
        axes.append(obj_key)

    # * choose database
    if (db == 'pf'):
        database = pf
        plotTitle = "Pareto Front"
    elif db == 'obj':
        database = obj_db
        plotTitle = "Objective Data"
    else:
        message = "'" + str(db) + "' is not an acceptible value for 'db'\n"
        message += "Consider using 'pf' or 'obj' instead."
        raise ValueError(message)

    hoverInfo = []
    for key in database.dtype.names:
        redundant = False
        for ax in axes:
            if ax == key:
                redundant = True
        if redundant is False:
            hoverInfo.append(key)

    # * create plot
    obj_fig = px.parallel_coordinates(database,
                                      labels=axes,
                                      title=plotTitle,
                                    #   hover_data=hoverInfo
                                      )

    # * display plot
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

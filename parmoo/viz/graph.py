""" This module contains tools to generate interactive Plotly-based graphs.

The functions are:

 Display data in interactive browser plot
  * ``generate_scatter(moop)`` -- Generate interactive scatterplot matrix
  * ``generate_parallel(moop)`` -- Generate interactive parallel plot
  * ``generate_radar(moop)`` -- Generate interactive radar plot

"""

import plotly.express as px
import plotly.graph_objects as go
from warnings import warn
from .utilities import setPlotName, setDatabase, setHoverInfo


#
# ! THESE FUNCTIONS DISPLAY DATA IN AN INTERACTIVE BROWSER PLOT
#


def generate_scatter(moop,
                     db,
                     height,
                     width,
                     verbose,):

    # * get info
    objectives = moop.getObjectiveType().names

    # * choose database
    database = setDatabase(moop, db=db)
    plotName = setPlotName(db=db)
    # * create plot
    if (len(objectives) == 2):
        fig = px.scatter(database,
                         x=objectives[0],
                         y=objectives[1],
                         title=plotName,
                         hover_data=database.columns)
        # fig.update_xaxes(showticklabels=False)
        # fig.update_yaxes(showticklabels=False)
    else:
        fig = px.scatter_matrix(database,
                                dimensions=objectives,
                                title=plotName,
                                hover_data=database.columns)
        fig.update_traces(diagonal_visible=False)

    # * return figure
    return fig


def generate_parallel(moop,
                      db,
                      height,
                      width,
                      verbose,
                      objectives_only,):

    # * setup axes
    objectives = moop.getObjectiveType().names
    if moop.getConstraintType() is not None:
        constraints = moop.getConstraintType().names
    else:
        constraints = ()

    # * choose database
    database = setDatabase(moop, db=db)
    plotName = setPlotName(db=db)

    # * create plot
    if objectives_only:
        fig = px.parallel_coordinates(database,
                                      dimensions=objectives,
                                      title=plotName,)
    else:
        axes = objectives + constraints
        fig = px.parallel_coordinates(database,
                                      labels=axes,
                                      title=plotName,)

    # * return figure
    return fig


def generate_radar(moop,
                   db,
                   height,
                   width,
                   verbose,):

    # * setup axes
    objectives = moop.getObjectiveType().names
    wrap = objectives[0]
    temp_variable = list(objectives)
    temp_variable.append(wrap)
    axes = tuple(temp_variable)

    # * choose database
    database = setDatabase(moop, db=db)
    plotName = setPlotName(db=db)

    # * create scaled database
    j = database.copy(deep=True)
    for i in j.columns:
        j[i] = (j[i] - j[i].min()) / (j[i].max() - j[i].min())
    scaled_db = j

    # * raise warnings
    if len(objectives) < 3:
        message = """
        Radar plots are best suited for MOOPs with at least 3 objectives.
        Consider using a scatterplot or parallel coordinates plot instead.
        """
        warn(message)

    # * create plot
    fig = go.Figure()
    for i in range(len(database)):
        trace = (i)
        hoverInfo = setHoverInfo(database=database, i=i,)
        values = []
        for key in axes:
            values.append(scaled_db[key][i])
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=axes,
            name=trace,
            hovertext=hoverInfo,))

    # * improve aesthetics
    fig.update_traces(
        hoverinfo='text',
        selector=dict(
            type='scatterpolar'))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showticklabels=False,)))
    fig.update_layout(
        title=dict(
            text=plotName))
    fig.update_layout(
        autosize=True,)

    # * return figure
    return fig

""" This module contains tools to generate interactive Plotly-based graphs.

The functions are:

 Display data in interactive browser plot
  * ``generate_scatter(moop)`` -- Generate interactive scatterplot matrix
  * ``generate_parallel(moop)`` -- Generate interactive parallel plot
  * ``generate_radar(moop)`` -- Generate interactive radar plot

"""

import plotly.express as px
import plotly.graph_objects as go
import logging
from .utilities import (
    set_plot_name,
    set_database,
    set_hover_info,
)


def generate_scatter(
    moop,
    db,
    points,
):

    """ Generate a scatterplot or scatterplot matrix.

        Args:
            moop (MOOP): An object containing the data to be visualized.

            db (string): Filter traces by dataset.
                'pf' - Plot the Pareto Front.
                'obj' - Plot objective data.

            points (string): Filter traces by constraint score.
                'constraint_satisfying' - Show only points that
                    satisfy every constraint.
                'constraint_violating' - Show only points that
                    violate any constraint.
                'all' - Plot all points.
                'none' - Plot no points.

        Returns:
            plotly.graph_objects.Figure: A scatterplot or scatterplot matrix
                displaying traces that fit the filtering criteria.
    """

    # * intro log
    logging.info('generating scatterplot. this might take a while')

    # * get info
    objectives = moop.getObjectiveType().names

    # * choose database
    database = set_database(moop, db=db, points=points)
    plot_name = set_plot_name(db=db)

    # * create plot
    if (len(objectives) == 2):
        fig = px.scatter(
            database,
            x=objectives[0],
            y=objectives[1],
            title=plot_name,
            hover_data=database.columns,
            template='none',
        )
        # fig.update_xaxes(showticklabels=False)
        # fig.update_yaxes(showticklabels=False)
    else:
        fig = px.scatter_matrix(
            database,
            dimensions=objectives,
            title=plot_name,
            hover_data=database.columns,
            template='none',
        )
        fig.update_traces(diagonal_visible=False)

    # * logging outro
    logging.info('generated scatterplot')

    # * return figure
    return fig


def generate_parallel(
    moop,
    db,
    points,
):

    """ Generate a parallel coordinates plot.

        Args:
            moop (MOOP): An object containing the data to be visualized.

            db (string): Filter traces by dataset.
                'pf' - Plot the Pareto Front.
                'obj' - Plot objective data.

            points (string): Filter traces by constraint score.
                'constraint_satisfying' - Show only points that
                    satisfy every constraint.
                'constraint_violating' - Show only points that
                    violate any constraint.
                'all' - Plot all points.
                'none' - Plot no points.

        Returns:
            plotly.graph_objects.Figure: A parallel coordinates plot
                displaying traces that fit the filtering criteria.
    """

    # * intro log
    message = 'generating parallel coordinates plot. '
    message += 'this might take a while'
    logging.info(message)

    # * setup axes
    objectives = moop.getObjectiveType().names

    # * choose database
    database = set_database(moop, db=db, points=points)
    plot_name = set_plot_name(db=db)

    # * create plot
    fig = px.parallel_coordinates(
        database,
        dimensions=objectives,
        title=plot_name,
        template='none',
    )

    # * logging outro
    logging.info('generated parallel coordinates plot')

    # * return figure
    return fig


def generate_radar(
    moop,
    db,
    points,
):

    """ Generate a radar plot.

        Args:
            moop (MOOP): An object containing the data to be visualized.

            db (string): Filter traces by dataset.
                'pf' - Plot the Pareto Front.
                'obj' - Plot objective data.

            points (string): Filter traces by constraint score.
                'constraint_satisfying' - Show only points that
                    satisfy every constraint.
                'constraint_violating' - Show only points that
                    violate any constraint.
                'all' - Plot all points.
                'none' - Plot no points.

        Returns:
            plotly.graph_objects.Figure: A radar plot
                displaying traces that fit the filtering criteria.
    """

    # * intro log
    logging.info('generating radar plot. this might take awhile')

    # * setup axes
    objectives = moop.getObjectiveType().names
    wrap = objectives[0]
    temp_variable = list(objectives)
    temp_variable.append(wrap)
    axes = tuple(temp_variable)

    # * choose database
    database = set_database(moop, db=db, points=points)
    plot_name = set_plot_name(db=db)

    # * create scaled database
    j = database.copy(deep=True)
    for i in j.columns:
        j[i] = (j[i] - j[i].min()) / (j[i].max() - j[i].min())
    scaled_db = j

    # * create plot
    fig = go.Figure()
    for i in range(len(database)):
        trace = (i)
        hover_info = set_hover_info(database=database, i=i,)
        values = []
        for key in axes:
            values.append(scaled_db[key][i])
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=axes,
            name=trace,
            hovertext=hover_info,))

    # * improve aesthetics
    fig.update_layout(
        template='none',
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showticklabels=False,)))
    fig.update_layout(
        title=dict(
            text=plot_name))
    fig.update_layout(
        autosize=True,)
    fig.update_layout(
        showlegend=False)

    # * logging outro
    logging.info('generated radar plot')

    # * return figure
    return fig

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

For all interactve browser plot functions, there is a known issue
where Plotly images may not export from Safari correctly. If you
encounter this issue, change your default browser to Chrome, Firefox,
or Edge.

"""

import plotly.express as px         # 15.2 MB package
import plotly.graph_objects as go
# import plotly.io as pio
# from parmoo import MOOP
# import numpy as np
from warnings import warn                    # native python package
from .dashboard import buildDashApp
import pandas as pd

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

# def printDataTypes(moop):
#     """ Print data types to terminal.

#     Args:
#         moop (MOOP): A ParMOO MOOP containing the data types to print.

#     Returns:
#         None

#     """
#     des_type = moop.getDesignType()
#     obj_type = moop.getObjectiveType()
#     sim_type = moop.getSimulationType()
#     const_type = moop.getConstraintType()
#     print("\nDATA TYPES FOR YOUR MOOP:\n")
#     print("Design variable type:   " + str(des_type))
#     print("Simulation output type: " + str(sim_type))
#     print("Objective type:         " + str(obj_type))
#     print("Constraint type:        " + str(const_type))


def scatter(moop,
            db='pf',
            output='dash',
            height='auto',
            width='auto',
            verbose=True,
            hot_reload=False,
            pop_up=True,
            port='http://127.0.0.1:8050/',):
    """ Display MOOP results as matrix of 2D scatterplots.

    Create an interactive plot that displays in the browser.

    For ``n`` objectives, generate an ``n x n`` matrix of 2D scatterplots.

    Users can hover above an output point to see input information pop up.

    Args:
        moop (MOOP): A ParMOO MOOP containing the MOOP results to plot.
        db (String): Choose database to plot
                     'pf' (default) plot Pareto Front
                     'obj' plot objective data
        output (String):
                     'dash' (default) display plot in dash app
                     'no_dash' display plot in browser without dash
                     'html' export plot as html to working directory
                     'pdf' export plot as pdf to working directory
                     'svg' export plot as svg to working directory
                     'webp' export plot as webp to working directory
                     'jpeg' export plot as jpeg to working directory
                     'png' export plot as png to working directory
        browser (boolean): Display interactive plot in browser window.
                    True: (default) display interactive plot in browser window
                    False: don't display interactive plot in browser window
                    It is recommended that this setting be left on True
                    The 'browser' and 'export' keywords will not
                    interfere with each other. If you choose to export an image
                    of the plot by using the 'export' keyword, and leave
                    'browser' to True, you will BOTH export an image file to
                    the current working directory AND open an interactive
                    figure in the browser

    Returns:
        None

    """

    # * get info
    obj_type = moop.getObjectiveType()

    # * choose axes
    axes = []  # each axis relates to an objective
    axCount = 0
    for obj_key in obj_type.names:
        axes.append(obj_key)
        axCount += 1

    # * choose database
    if (db == 'pf'):
        database = pd.DataFrame(moop.getPF())
        plotName = "Pareto Front"
    elif db == 'obj':
        database = pd.DataFrame(moop.getObjectiveData())
        plotName = "Objective Data"
    else:
        message = "'" + str(db) + "' is not an acceptible value for 'db'\n"
        message += "Consider using 'pf' or 'obj' instead."
        raise ValueError(message)

    # * set up hoverinfo
    hoverInfo = []
    for key in database.columns:
        redundant = False
        for ax in axes:
            if ax == key:
                redundant = True
        if redundant is False:
            hoverInfo.append(key)

    # * create plot
    if (axCount == 2):
        fig = px.scatter(database,
                         x=axes[0],
                         y=axes[1],
                         title=plotName,
                         hover_data=hoverInfo,
                         )
    else:
        fig = px.scatter_matrix(database,
                                dimensions=axes,
                                title=plotName,
                                hover_data=hoverInfo,
                                )
        fig.update_traces(diagonal_visible=False)

    # * configure plot
    config = configure(export=output,
                       height=height,
                       width=width,
                       plotName=plotName,)

    # * output
    if output == 'dash':
        buildDashApp(moop=moop,
                     db=db,
                     fig=fig,
                     config=config,
                     verbose=verbose,
                     hot_reload=hot_reload,
                     pop_up=pop_up,
                     port=port,)
    elif output == 'no_dash':
        fig.show(config=config)
    else:
        exportFile(fig=fig,
                   plotName=plotName,
                   fileType=output)


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


def radar(moop,
          db='pf',
          output='dash',
          height='auto',
          width='auto',
          verbose=True,
          hot_reload=False,
          pop_up=True,
          port='http://127.0.0.1:8050/',):
    """ Display MOOP results as radar plot.

    Create an interactive plot that displays in the browser.

    Args:
        moop (MOOP): A ParMOO MOOP containing the results to plot.
        db (String): Choose database to plot
                     'pf' (default) plot Pareto Front
                     'obj' plot objective data
        export (String): Export plot to working directory.
                     'none' (default) don't export image file
                     'html' export plot as html
                     'pdf' export plot as pdf
                     'svg' export plot as svg
                     'webp' export plot as webp
                     'jpeg' export plot as jpeg
                     'png' export plot as png
        browser (boolean): Display interactive plot in browser window.
                    True: (default) display interactive plot in browser window
                    False: don't display interactive plot in browser window
                    It is recommended that this setting be left on True
                    The 'browser' and 'export' keywords will not
                    interfere with each other. If you choose to export an image
                    of the plot by using the 'export' keyword, and leave
                    'browser' to True, you will BOTH export an image file to
                    the current working directory AND open an interactive
                    figure in the browser.

    Returns:
        None

    """

    # * setup axes
    axes = []
    obj_type = moop.getObjectiveType()
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
        database = pd.DataFrame(moop.getPF())
        plotName = "Pareto Front"
    elif db == 'obj':
        database = pd.DataFrame(moop.getObjectiveData())
        plotName = "Objective Data"
    else:
        message = "'" + str(db) + "' is not an acceptible value for 'db'\n"
        message += "Consider using 'pf' or 'obj' instead."
        raise ValueError(message)

    # * create scaled database
    j = pd.DataFrame(database)
    for i in j.columns:
        j[i] = (j[i] - j[i].min()) / (j[i].max() - j[i].min())
    scaled_db = j

    # * raise warnings
    if len(axes) < 3:
        message = """
        Radar plots are best suited for MOOPs with at least 3 objectives.
        Consider using a scatterplot or parallel coordinates plot instead.
        """
        warn(message)

    # * configure plot
    config = configure(export=output,
                       height=height,
                       width=width,
                       plotName=plotName,)

    # * create plot
    fig = go.Figure()
    for i in range(len(database)):
        traceName = (i)
        if plotName == "Pareto Front":
            hoverInfo = ""
        else:
            hoverInfo = "Design #" + str(i) + "\n"
        for key in database.columns:
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
            values.append(scaled_db[obj_key][i])
            if count == 0:
                wrap_around_value = obj_key
            count += 1
        values.append(scaled_db[wrap_around_value][i])
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=axes,
            name=traceName,
            hovertext=hoverInfo,
        ))

    # * improve aesthetics
    fig.update_traces(
        hoverinfo='text',
        selector=dict(
            type='scatterpolar'
        )
    )
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
            text=plotName
        )
    )
    fig.update_layout(
        autosize=True,
    )
    if plotName == "Pareto Front":
        fig.update_layout(showlegend=True)
    else:
        fig.update_layout(showlegend=True)

    # * output
    if output == 'dash':
        buildDashApp(moop=moop,
                     db=db,
                     fig=fig,
                     config=config,
                     verbose=verbose,
                     hot_reload=hot_reload,
                     pop_up=pop_up,
                     port=port,)
    elif output == 'no_dash':
        fig.show(config=config)
    else:
        exportFile(fig=fig,
                   plotName=plotName,
                   fileType=output)


def parallel_coordinates(moop,
                         db='pf',
                         output='dash',
                         height='auto',
                         width='auto',
                         objectives_only=True,
                         verbose=True,
                         hot_reload=False,
                         pop_up=True,
                         port='http://127.0.0.1:8050/',):
    """ Display MOOP results as parallel coordinates plot.

    Create an interactive plot that displays in the display.

    Users can select item(s) in a parallel coordinates plot
    by selecting an axis section which item(s) pass through.

    Args:
        moop (MOOP): A ParMOO MOOP containing the results to plot.
        db (String): Choose database to plot
                     'pf' (default) plot Pareto Front
                     'obj' plot objective data
        export (String): Export plot to working directory.
                     'none' (default) don't export image file
                     'html' export plot as html
                     'pdf' export plot as pdf
                     'svg' export plot as svg
                     'webp' export plot as webp
                     'jpeg' export plot as jpeg
                     'png' export plot as png
        browser (boolean): Display interactive plot in browser window.
                    True: (default) display interactive plot in browser window
                    False: don't display interactive plot in browser window
                    It is recommended that this setting be left on True
                    The 'browser' and 'export' keywords will not
                    interfere with each other. If you choose to export an image
                    of the plot by using the 'export' keyword, and leave
                    'browser' to True, you will BOTH export an image file to
                    the current working directory AND open an interactive
                    figure in the browser.
        objectives_only (boolean): display all data, or objectives only
                    True: (default) plot objectives as axes only
                    False: plot inputs as axes as well

    Returns:
        None

    """

    # * setup axes
    axes = []  # each axis relates to an objective
    obj_type = moop.getObjectiveType()
    for obj_key in obj_type.names:
        axes.append(obj_key)
    if db == 'obj':
        const_type = moop.getConstraintType()
        if const_type is not None:
            for const_key in const_type.names:
                if const_key is not None:
                    axes.append(const_key)

    # * choose database
    if (db == 'pf'):
        database = pd.DataFrame(moop.getPF())
        plotName = "Pareto Front"
    elif db == 'obj':
        database = pd.DataFrame(moop.getObjectiveData())
        plotName = "Objective Data"
    else:
        message = "'" + str(db) + "' is not an acceptible value for 'db'\n"
        message += "Consider using 'pf' or 'obj' instead."
        raise ValueError(message)

    # * create plot
    if objectives_only:
        fig = px.parallel_coordinates(database,
                                      dimensions=axes,
                                      title=plotName,)
    else:
        fig = px.parallel_coordinates(database,
                                      labels=axes,
                                      title=plotName,)

    # * configure plot
    config = configure(export=output,
                       height=height,
                       width=width,
                       plotName=plotName,)

    # * output
    if output == 'dash':
        buildDashApp(moop=moop,
                     db=db,
                     fig=fig,
                     config=config,
                     verbose=verbose,
                     hot_reload=hot_reload,
                     pop_up=pop_up,
                     port=port,)
    elif output == 'no_dash':
        fig.show(config=config)
    else:
        exportFile(fig=fig,
                   plotName=plotName,
                   fileType=output)


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


def exportFile(fig, plotName, fileType):
    """ Display MOOP plot.

    Export plot.

    Args:
        fig (Plotly figure): figure to export.
        plotName (String): Used for naming export files
        export (String): Indicate export type
                     'none' (default) don't export image file
                     'html' export plot as html
                     'pdf' export plot as pdf
                     'svg' export plot as svg
                     'webp' export plot as webp
                     'jpeg' export plot as jpeg
                     'png' export plot as png

    Returns:
        None

    """
    if fileType == 'html':
        fig.write_html(plotName + ".html")
    elif fileType == 'pdf':
        fig.write_image(plotName + ".pdf")
    elif fileType == 'svg':
        fig.write_image(plotName + ".svg")
    elif fileType == 'webp':
        fig.write_image(plotName + ".webp")
    elif fileType == 'jpeg':
        fig.write_image(plotName + ".jpeg")
    elif fileType == 'png':
        fig.write_image(plotName + ".png")
    else:
        message = "ParMOO does not support outputting to '" + fileType + "'.\n"
        message += "Supported outputs:\n"
        message += "'dash'\n"
        message += "'no_dash'\n"
        message += "'html'\n"
        message += "'pdf'\n"
        message += "'svg'\n"
        message += "'webp'\n"
        message += "'jpeg'\n"
        message += "'png'\n"
        raise ValueError(message)


def configure(export, height, width, plotName):

    # # * set screenshot type based on export type
    # if export == 'png':
    #     screenshot = export
    # elif export == 'webp':
    #     screenshot = export
    # elif export == 'jpeg':
    #     screenshot = export
    # elif export
    #     screenshot = 'svg'
    screenshot = 'svg'

    # * set config based on scale
    if height != 'auto' and width != 'auto':
        config = {
            'displaylogo': False,
            'displayModeBar': True,
            'toImageButtonOptions': {
                'format': screenshot,  # one of png, svg, jpeg, webp
                'filename': str(plotName),
                'height': int(height),
                'width': int(width),
                'scale': 1  # Multiply title/legend/axis/canvas sizes by factor
            }
        }
    else:
        config = {
            'displaylogo': False,
            'displayModeBar': True,
            'toImageButtonOptions': {
                'format': screenshot,  # one of png, svg, jpeg, webp
                'filename': str(plotName),
            }
        }
    return config

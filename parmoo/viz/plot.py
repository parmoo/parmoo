import logging
from .dashboard import (
    Dash_App
)
from .utilities import (
    export_file,
    set_plot_name,
    check_inputs,
)
from .graph import (
    generate_scatter,
    generate_parallel,
    generate_radar,
)

"""
For all interactve browser plot functions, there is a known issue
where Plotly images may not export from Safari correctly. If you
encounter this issue, change your default browser to Chrome, Firefox,
or Edge.
"""

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


def scatter(
    moop,
    db='pf',
    output='dash',
    points='constraint_satisfying',
    height='auto',
    width='auto',
    font='auto',
    fontsize='auto',
    background_color='auto',
    screenshot='svg',
    image_export_format='svg',
    data_export_format='csv',
    dev_mode=False,
    pop_up=True,
    port='http://127.0.0.1:8050/',
):
    check_inputs(
        db=db,
        output=output,
        points=points,
        height=height,
        width=width,
        font=font,
        fontsize=fontsize,
        background_color=background_color,
        screenshot=screenshot,
        image_export_format=image_export_format,
        data_export_format=data_export_format,
        dev_mode=dev_mode,
        pop_up=pop_up,
        port=port,
    )
    logging.info('initialized scatter() wrapper')
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
        Dash_App(
            plot_type='scatter',
            moop=moop,
            db=db,
            points=points,
            height=height,
            width=width,
            font=font,
            fontsize=fontsize,
            background_color=background_color,
            screenshot=screenshot,
            image_export_format=image_export_format,
            data_export_format=data_export_format,
            dev_mode=dev_mode,
            pop_up=pop_up,
            port=port,
        )
    else:
        plot_name = set_plot_name(db=db)
        export_file(
            fig=generate_scatter(
                moop,
                db=db,
                points=points,
            ),
            plot_name=plot_name,
            file_type=output,
        )


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


def parallel_coordinates(
    moop,
    db='pf',
    output='dash',
    points='constraint_satisfying',
    height='auto',
    width='auto',
    font='auto',
    fontsize='auto',
    background_color='auto',
    screenshot='svg',
    image_export_format='svg',
    data_export_format='csv',
    dev_mode=False,
    pop_up=True,
    port='http://127.0.0.1:8050/',
):
    check_inputs(
        db=db,
        output=output,
        points=points,
        height=height,
        width=width,
        font=font,
        fontsize=fontsize,
        background_color=background_color,
        screenshot=screenshot,
        image_export_format=image_export_format,
        data_export_format=data_export_format,
        dev_mode=dev_mode,
        pop_up=pop_up,
        port=port,
    )
    logging.info('initialized parallel_coordinates() wrapper')
    if output == 'dash':
        Dash_App(
            plot_type='parallel',
            moop=moop,
            db=db,
            points=points,
            height=height,
            width=width,
            font=font,
            fontsize=fontsize,
            background_color=background_color,
            screenshot=screenshot,
            image_export_format=image_export_format,
            data_export_format=data_export_format,
            dev_mode=dev_mode,
            pop_up=pop_up,
            port=port,
        )
    else:
        plot_name = set_plot_name(db=db)
        export_file(
            fig=generate_parallel(
                moop=moop,
                db=db,
                points=points,
            ),
            plot_name=plot_name,
            file_type=output
        )


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


def radar(
    moop,
    db='pf',
    output='dash',
    points='constraint_satisfying',
    height='auto',
    width='auto',
    font='auto',
    fontsize='auto',
    background_color='auto',
    screenshot='svg',
    image_export_format='svg',
    data_export_format='csv',
    dev_mode=False,
    pop_up=True,
    port='http://127.0.0.1:8050/',
):
    check_inputs(
        db=db,
        output=output,
        points=points,
        height=height,
        width=width,
        font=font,
        fontsize=fontsize,
        background_color=background_color,
        screenshot=screenshot,
        image_export_format=image_export_format,
        data_export_format=data_export_format,
        dev_mode=dev_mode,
        pop_up=pop_up,
        port=port,
    )
    logging.info('initialized radar() wrapper')
    if output == 'dash':
        Dash_App(
            plot_type='radar',
            moop=moop,
            db=db,
            points=points,
            height=height,
            width=width,
            font=font,
            fontsize=fontsize,
            background_color=background_color,
            screenshot=screenshot,
            image_export_format=image_export_format,
            data_export_format=data_export_format,
            dev_mode=dev_mode,
            pop_up=pop_up,
            port=port,
        )
    else:
        plot_name = set_plot_name(db=db)
        export_file(
            fig=generate_radar(
                moop,
                db=db,
                points=points,
            ),
            plot_name=plot_name,
            file_type=output
        )

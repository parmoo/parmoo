
""" A plotting library for interactive visualization of MOOP objects.

To generate a plot, generate create a MOOP object and use it to solve a
problem. Then call one of the following functions, passing the MOOP object
as the first argument:

 * ``viz.scatter(moop)``
 * ``viz.parallel_coordinates(moop)``
 * ``viz.radar(moop)``

Please take note of the following:

 * Via the interactivity features, all of the arguments except for the
   ``moop`` field can be adjusted interactively through the pop-up GUI.
   The plot type can also be adjusted from the dropdown menu while running
   the GUI.
 * The viz tool hosts a Dash app through the local port
   ``http://127.0.0.1:8050/``.
   This means that only one visualization can be hosted at a time.
   Also, killing the Dash app will end the interactivity in the browser,
   although a static plot may remain in your browser window.
 * Finally, note that there is a known issue when using the Chrome browser.
   The Firefox or Safari browsers are recommended.

To interact with the plots:

 1. To multi-select in scatterplots and radar plots, hold down ``SHIFT``
    while making additional selections.
 2. To remove all selections from a scatterplot and radar plot, select without
    holding down the ``SHIFT`` key or double click.
 3. To select in parallel coordinates plots, click along an axis and then
    drag the cursor elsewhere along the same axis.
    The highlighted part of the axis is the area that is selected.
    There is no need to hold down SHIFT when making multiple selections in
    parallel coordinates plots.
 4. To remove a parallel coordinates plot selection, click on the highlighted
    selection bar you wish to delete.
 5. To reset plot interactions made through the toolbar in the top right of a
    graph (not interactions made through buttons, dropdowns, toggles, or input
    boxes), double click on the plot. This will undo selections, zoom, pan,
    etc.

The three basic plot options are detailed below.

"""

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


def scatter(moop,
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
    """ Create a scatter plot matrix to visualize the results of a MOOP.

    Args:
        moop (MOOP): The MOOP results that you would like to visualize.

        db (str): Either 'pf' to plot just the Pareto front, or 'obj'
            to plot the complete objective database. Defaults to 'pf'.

        output (str): Either 'dash' to generate an interactive plot
            running in your browser using the dash app, or anything
            else to save a static plot to the desktop. Defaults to
            'dash'.

        points (str): Plot only
            constraint satisfying points ('constraint_satisfying'),
            constraint violating points ('constraint_violating'),
            all points ('all'),
            or no points ('none').

        height (str): The height in pixels of the resulting figure.
            Defaults to 'auto', which matches your screen size.

        width (str): The width in pixels of the resulting figure.
            Defaults to 'auto', which matches your screen size.

        font (str): The font that will be used for axis labels and legends.
            These values are automatically inferred from the name fields
            of your MOOP object. Any specified font must be available on your
            computer and available in the appropriate path. Defaults 'auto',
            which is times new roman on most machines.

        fontsize (str): The font size (in points). Defaults to 'auto', which
            infers the size based on the plot dimensions.

        background_color (str): Set the background color for this plot.
            Defaults to 'auto', which is white with grey axis lines on
            most systems.

        screenshot (str): Set the download mode when saving a screenshot
            using the "screenshot" button. Defaults to 'svg'.
            Other available options include:
            'html', 'webp', 'jpeg', 'png', 'svg', 'eps', and 'pdf'.
            Note that the 'eps' option requires the poppler library, which
            is not included in any of ParMOO's dependency lists.

        image_export_format (str): Set the export format when exporting
            a plot directly image file. Defaults to 'svg'.
            Other available options include:
            'html', 'webp', 'jpeg', 'png', 'svg', 'eps', and 'pdf'.
            Note that the 'eps' option requires the poppler library, which
            is not included in any of ParMOO's dependency lists.

        data_export_format (str): Set the format for exporting selected data
            to a file. Defaults to 'csv'. The other option is 'json'.

        dev_mode (bool): Run in developer mode, which allows changes to the
            code to automatically render in the browser. Activating this
            mode will interfere with some functionalities (such as
            checkpointing) since it results in multiple calls to the script.
            This value defaults to False, and should only be adjusted by
            developers.

        pop_up (bool): Automatically pop-up the dash app when called.
            Defaults to True. The only reason one might want to adjust, is
            if the environment prevents pop-ups or a non default browser
            is desired.

        port (str): The port through which the Dash app is hosted.
            Defaults to 'http://127.0.0.1:8050/'.

    """

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


def parallel_coordinates(moop,
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
                         port='http://127.0.0.1:8050/'
                         ):
    """ Create a parallel coordinates plot to visualize the results of a MOOP.

    Args:
        moop (MOOP): The MOOP results that you would like to visualize.

        db (str): Either 'pf' to plot just the Pareto front, or 'obj'
            to plot the complete objective database. Defaults to 'pf'.

        output (str): Either 'dash' to generate an interactive plot
            running in your browser using the dash app, or anything
            else to save a static plot to the desktop. Defaults to
            'dash'.

        points (str): Plot only
            constraint satisfying points ('constraint_satisfying'),
            constraint violating points ('constraint_violating'),
            all points ('all'),
            or no points ('none').

        height (str): The height in pixels of the resulting figure.
            Defaults to 'auto', which matches your screen size.

        width (str): The width in pixels of the resulting figure.
            Defaults to 'auto', which matches your screen size.

        font (str): The font that will be used for axis labels and legends.
            These values are automatically inferred from the name fields
            of your MOOP object. Any specified font must be available on your
            computer and available in the appropriate path. Defaults 'auto',
            which is times new roman on most machines.

        fontsize (str): The font size (in points). Defaults to 'auto', which
            infers the size based on the plot dimensions.

        background_color (str): Set the background color for this plot.
            Defaults to 'auto', which is white with grey axis lines on
            most systems.

        screenshot (str): Set the download mode when saving a screenshot
            using the "screenshot" button. Defaults to 'svg'.
            Other available options include:
            'html', 'webp', 'jpeg', 'png', 'svg', 'eps', and 'pdf'.
            Note that the 'eps' option requires the poppler library, which
            is not included in any of ParMOO's dependency lists.

        image_export_format (str): Set the export format when exporting
            a plot directly image file. Defaults to 'svg'.
            Other available options include:
            'html', 'webp', 'jpeg', 'png', 'svg', 'eps', and 'pdf'.
            Note that the 'eps' option requires the poppler library, which
            is not included in any of ParMOO's dependency lists.

        data_export_format (str): Set the format for exporting selected data
            to a file. Defaults to 'csv'. The other option is 'json'.

        dev_mode (bool): Run in developer mode, which allows changes to the
            code to automatically render in the browser. Activating this
            mode will interfere with some functionalities (such as
            checkpointing) since it results in multiple calls to the script.
            This value defaults to False, and should only be adjusted by
            developers.

        pop_up (bool): Automatically pop-up the dash app when called.
            Defaults to True. The only reason one might want to adjust, is
            if the environment prevents pop-ups or a non default browser
            is desired.

        port (str): The port through which the Dash app is hosted.
            Defaults to 'http://127.0.0.1:8050/'.

    """

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
    """ Create a radar plot to visualize the results of a MOOP.

    Args:
        moop (MOOP): The MOOP results that you would like to visualize.

        db (str): Either 'pf' to plot just the Pareto front, or 'obj'
            to plot the complete objective database. Defaults to 'pf'.

        output (str): Either 'dash' to generate an interactive plot
            running in your browser using the dash app, or anything
            else to save a static plot to the desktop. Defaults to
            'dash'.

        points (str): Plot only
            constraint satisfying points ('constraint_satisfying'),
            constraint violating points ('constraint_violating'),
            all points ('all'),
            or no points ('none').

        height (str): The height in pixels of the resulting figure.
            Defaults to 'auto', which matches your screen size.

        width (str): The width in pixels of the resulting figure.
            Defaults to 'auto', which matches your screen size.

        font (str): The font that will be used for axis labels and legends.
            These values are automatically inferred from the name fields
            of your MOOP object. Any specified font must be available on your
            computer and available in the appropriate path. Defaults 'auto',
            which is times new roman on most machines.

        fontsize (str): The font size (in points). Defaults to 'auto', which
            infers the size based on the plot dimensions.

        background_color (str): Set the background color for this plot.
            Defaults to 'auto', which is white with grey axis lines on
            most systems.

        screenshot (str): Set the download mode when saving a screenshot
            using the "screenshot" button. Defaults to 'svg'.
            Other available options include:
            'html', 'webp', 'jpeg', 'png', 'svg', 'eps', and 'pdf'.
            Note that the 'eps' option requires the poppler library, which
            is not included in any of ParMOO's dependency lists.

        image_export_format (str): Set the export format when exporting
            a plot directly image file. Defaults to 'svg'.
            Other available options include:
            'html', 'webp', 'jpeg', 'png', 'svg', 'eps', and 'pdf'.
            Note that the 'eps' option requires the poppler library, which
            is not included in any of ParMOO's dependency lists.

        data_export_format (str): Set the format for exporting selected data
            to a file. Defaults to 'csv'. The other option is 'json'.

        dev_mode (bool): Run in developer mode, which allows changes to the
            code to automatically render in the browser. Activating this
            mode will interfere with some functionalities (such as
            checkpointing) since it results in multiple calls to the script.
            This value defaults to False, and should only be adjusted by
            developers.

        pop_up (bool): Automatically pop-up the dash app when called.
            Defaults to True. The only reason one might want to adjust, is
            if the environment prevents pop-ups or a non default browser
            is desired.

        port (str): The port through which the Dash app is hosted.
            Defaults to 'http://127.0.0.1:8050/'.

    """

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

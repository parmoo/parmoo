""" This module contains utilities (helper functions) that are used throughout
the viz tool.

"""

import pandas as pd
import logging


def export_file(fig, plot_name, file_type):
    """ Export image of figure to working directory.

        Args:
            fig (plotly.graph_objects.Figure): The figure to export.

            plot_name (string): Set the filename of the image file.

            file_type (string): Set the image file type.
             - 'html' - Export as .html file.
             - 'pdf' - Export as .pdf file.
             - 'svg' - Export as .svg file.
             - 'eps' - Export as .eps file
               if the poppler dependency is installed.
             - 'jpeg' - Export as .jpeg file.
             - 'png' - Export as .png file.
             - 'webp' - Export as .webp file.

    """

    if file_type == 'html':
        fig.write_html(plot_name + ".html")
        logging.info("exported graph as .html")
    elif file_type == 'pdf':
        fig.write_image(plot_name + ".pdf")
        logging.info("exported graph as .pdf")
    elif file_type == 'svg':
        fig.write_image(plot_name + ".svg")
        logging.info("exported graph as .svg")
    elif file_type == 'eps':
        fig.write_image(plot_name + ".eps")
        logging.info("exported graph as .eps")
    elif file_type == 'jpeg':
        fig.write_image(plot_name + ".jpeg")
        logging.info("exported graph as .jpeg")
    elif file_type == 'png':
        fig.write_image(plot_name + ".png")
        logging.info("exported graph as .png")
    elif file_type == 'webp':
        fig.write_image(plot_name + ".webp")
        logging.info("exported graph as .webp")


def set_plot_name(db):
    """ Provide a default graph title.

        Args:
            db (string): Graph contents inform title.
             - 'pf' - Set plot name to "Pareto Front"
             - 'obj' - Set plot name to "Objective Data"

        Returns:
            plot_name (string): The default plot name.

    """

    if db == 'pf':
        plot_name = "Pareto Front"
    elif db == 'obj':
        plot_name = "Objective Data"
    return plot_name


def set_database(moop, db, points):
    """ Choose which points from MOOP object to plot.

        Args:
            db (string): Set dataset.
             - 'pf' - Set Pareto Front as dataset.
             - 'obj' - Set objective data as dataset.

            points (string): Filter traces from dataset by constraint score.
             - 'constraint_satisfying' - Include only points that
               satisfy every constraint.
             - 'constraint_violating' - Include only points that
               violate any constraint.
             - 'all' - Include all points in dataset.
             - 'none' - Include no points in dataset.

        Returns:
            df (Pandas dataframe): A 2D dataframe containing post-filter
            data from the MOOP.

    """

    if db == 'pf':
        database = pd.DataFrame(moop.getPF())
    elif db == 'obj':
        database = pd.DataFrame(moop.getObjectiveData())
    if moop.getConstraintType() is None:
        df = database
    else:
        if points == 'constraint_satisfying':
            constraints = moop.getConstraintType().names
            df = database.copy(deep=True)
            for constraint in constraints:
                indices = df[df[constraint] > 0].index
                df.drop(indices, inplace=True)
                df.reset_index(inplace=True, drop=True)
        elif points == 'constraint_violating':
            constraints = moop.getConstraintType().names
            df = database.copy(deep=True)
            for constraint in constraints:
                indices = df[df[constraint] <= 0].index
                df.drop(indices, inplace=True)
                df.reset_index(inplace=True, drop=True)
        elif points == 'all':
            df = database
        elif points == 'none':
            df = database[0:0]
    return df


def set_hover_info(database, i):
    """ Customize information in hover label for trace i.

        Args:
            database (Pandas dataframe): A 2D dataframe containing the
                traces to be graphed.

            i (int): An index indicating the row where the trace
                we're labeling is located.

        Returns:
            hover_info (string): An HTML-format string to display when
            users hover over trace i.

    """

    hover_info = ""
    for key in database.columns:
        hover_info += str(key)
        hover_info += ": "
        hover_info += str(database[key][i])
        hover_info += "<br>"
        # since plotly is JavaScript-based
        # it uses HTML string formatting
    return hover_info


def check_inputs(db,
                 output,
                 points,
                 height,
                 width,
                 font,
                 fontsize,
                 background_color,
                 screenshot,
                 image_export_format,
                 data_export_format,
                 dev_mode,
                 pop_up,
                 port,
                 ):
    """ Check keyword inputs to user-facing functions for validity

        Args:
            db: The item passed to the 'db' keyword in a user-facing function.
                If db cannot be cast to a string valued 'pf' or 'obj',
                a ValueError is raised.

            output: The item passed to the 'output' keyword in a
                user-facing function.
                If output cannot be cast to a string corresponding to one of
                the supported output filetypes, a ValueError is raised.

            points: The item passed to the 'points' keyword in a
                user-facing function.
                If points cannot be cast to a string corresponding to one of
                the supported constraint filters, a ValueError is raised.

            height: The item passed to the 'height' keyword in a user-facing
                function.
                If height is not the default string 'auto' or cannot be cast
                to an int
                of value greater than one, a ValueError is raised.

            width: The item passed to the 'width' keyword in a user-facing
                function.
                If width is not the default string 'auto' or cannot be cast
                to an int
                of value greater than one, a ValueError is raised.

            font: The item passed to the 'font' keyword in a user-facing
                function.
                If font cannot be cast to a string, a ValueError is raised.

            fontsize: The item passed to the 'fontsize' keyword in a
                user-facing function.
                If fontsize is not the default value 'auto' or cannot be cast
                to an int
                of value between 1 and 100 inclusive, a ValueError is raised.

            background_color: The item passed to the 'background_color'
                keyword in a user-facing function.
                If background_color cannot be cast to a string, a ValueError
                is raised.

            screenshot: The item passed to the 'screenshot' keyword in a
                user-facing function.
                If screenshot cannot be cast to a string corresponding to one
                of the supported
                screenshot filetypes, a ValueError is raised.

            image_export_format: The item passed to the 'image_export_format'
                keyword in a user-facing function.
                If image_export_format cannot be cast to a string
                corresponding to one of the supported
                image_export_format filetypes, a ValueError is raised.

            data_export_format: The item passed to the 'data_export_format'
                keyword in a user-facing function.
                If data_export_format cannot be cast to a string corresponding
                to one of the supported
                data_export_format filetypes, a ValueError is raised.

            data_export_format: The item passed to the 'data_export_format'
                keyword in a user-facing function.
                If data_export_format cannot be cast to a string corresponding
                to one of the supported
                data_export_format filetypes, a ValueError is raised.

            dev_mode: The item passed to the 'dev_mode' keyword in a
                user-facing function.
                If dev_mode cannot be cast to one of the Boolean values True
                and False, a ValueError is raised.

            pop_up: The item passed to the 'pop_up' keyword in a user-facing
                function.
                If pop_up cannot be cast to one of the Boolean values True and
                False, a ValueError is raised.

            port: The item passed to the 'port' keyword in a user-facing
                function.
                If port cannot be cast to a string beginning with 'http', a
                ValueError is raised.

        Raises:
            A ValueError if any of the values passed by a user to a keyword in
            a user-facing function are judged invalid.

    """

    try:
        if (str(db) == 'pf' or str(db) == 'obj'):
            pass
        else:
            raise ValueError(str(db) + " is an invalid value for 'db'")
    except BaseException:
        raise ValueError(str(db) + " is an invalid value for 'db'")

    try:
        if (str(output) == 'dash' or
                str(output) == 'html' or
                str(output) == 'svg' or
                str(output) == 'pdf' or
                str(output) == 'eps' or
                str(output) == 'jpeg' or
                str(output) == 'png' or
                str(output) == 'webp'):
            pass
        else:
            raise ValueError(str(output) + " is an invalid value for 'output'")
    except BaseException:
        raise ValueError(str(output) + " is an invalid value for 'output'")

    try:
        if (str(points) == 'constraint_satisfying' or
                str(points) == 'constraint_violating' or
                str(points) == 'all' or
                str(points) == 'none'):
            pass
        else:
            raise ValueError(str(points) + " is an invalid value for 'points'")
    except BaseException:
        raise ValueError(str(points) + " is an invalid value for 'points'")

    if (height == 'auto' or int(height) >= 1):
        pass
    else:
        raise ValueError(str(height) + " is an invalid value for 'height'")

    try:
        if (width == 'auto' or int(width) >= 1):
            pass
        else:
            raise ValueError(str(width) + " is an invalid value for 'width'")
    except BaseException:
        raise ValueError(str(width) + " is an invalid value for 'width'")

    try:
        if (str(type(str(font))) == "<class 'str'>"):
            pass
        else:
            raise ValueError(str(font) + " is an invalid value for 'font'")
    except BaseException:
        raise ValueError(str(font) + " is an invalid value for 'font'")

    try:
        if (fontsize == 'auto' or
           (int(fontsize) >= 1 and int(fontsize) <= 100)):
            pass
        else:
            message = str(fontsize)
            message += " is an invalid value for 'fontsize'"
            raise ValueError(message)
    except BaseException:
        message = str(fontsize)
        message += " is an invalid value for 'fontsize'"
        raise ValueError(message)

    try:
        if (str(type(str(background_color))) == "<class 'str'>"):
            pass
        else:
            message = str(background_color)
            message += " is an invalid value for 'background_color'"
            raise ValueError(message)
    except BaseException:
        message = str(background_color)
        message += " is an invalid value for 'background_color'"
        raise ValueError(message)

    try:
        if (screenshot == 'png' or
                screenshot == 'svg' or
                screenshot == 'jpeg' or
                screenshot == 'webp'):
            pass
        else:
            message = str(screenshot)
            message += " is an invalid value for 'screenshot'"
            raise ValueError(message)
    except BaseException:
        message = str(screenshot)
        message += " is an invalid value for 'screenshot'"
        raise ValueError(message)

    try:
        if (image_export_format == 'html' or
                image_export_format == 'svg' or
                image_export_format == 'pdf' or
                image_export_format == 'eps' or
                image_export_format == 'jpeg' or
                image_export_format == 'png' or
                image_export_format == 'webp'):
            pass
        else:
            message = str(image_export_format)
            message += " is an invalid value for 'image_export_format'"
            raise ValueError(message)
    except BaseException:
        message = str(image_export_format)
        message += " is an invalid value for 'image_export_format'"
        raise ValueError(message)

    try:
        if (data_export_format == 'json' or
                data_export_format == 'csv'):
            pass
        else:
            message = str(data_export_format)
            message += " is an invalid value for 'data_export_format'"
            raise ValueError(message)
    except BaseException:
        message = str(data_export_format)
        message += " is an invalid value for 'data_export_format'"
        raise ValueError(message)

    try:
        if (dev_mode or not dev_mode):
            pass
        else:
            message = str(dev_mode)
            message += " is an invalid value for 'dev_mode'"
            raise ValueError(message)
    except BaseException:
        message = str(dev_mode)
        message += " is an invalid value for 'dev_mode'"
        raise ValueError(message)

    try:
        if (pop_up or not pop_up):
            pass
        else:
            raise ValueError(str(pop_up) + " is an invalid value for 'pop_up'")
    except BaseException:
        raise ValueError(str(pop_up) + " is an invalid value for 'pop_up'")

    try:
        if (port[0:4] == 'http'):
            pass
        else:
            raise ValueError(str(port) + " is an invalid value for 'port'")
    except BaseException:
        raise ValueError(str(port) + " is an invalid value for 'port'")

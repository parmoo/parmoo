import pandas as pd
import logging


def export_file(fig, plot_name, file_type):
    """ Display MOOP plot.

    Export plot.

    Args:
        fig (Plotly figure): figure to export.
        plot_name (String): Used for naming export files
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
    if file_type == 'html':
        fig.write_html(plot_name + ".html")
        logging.info("exported graph as .html")
    elif file_type == 'pdf':
        fig.write_image(plot_name + ".pdf")
        logging.info("exported graph as .pdf")
    elif file_type == 'svg':
        fig.write_image(plot_name + ".svg")
        logging.info("exported graph as .svg")
    elif file_type == 'webp':
        fig.write_image(plot_name + ".webp")
        logging.info("exported graph as .webp")
    elif file_type == 'jpeg':
        fig.write_image(plot_name + ".jpeg")
        logging.info("exported graph as .jpeg")
    elif file_type == 'png':
        fig.write_image(plot_name + ".png")
        logging.info("exported graph as .png")
    elif file_type == 'eps':
        fig.write_image(plot_name + ".eps")
        logging.info("exported graph as .eps")


def configure(height, width, plot_name, screenshot):

    # * set screenshot type
    screenshot = screenshot

    # * set config based on scale
    if height != 'auto' and width != 'auto':
        config = {
            'displaylogo': False,
            'displayModeBar': True,
            'toImageButtonOptions': {
                'format': screenshot,  # one of png, svg, jpeg, webp
                'filename': str(plot_name),
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
                'filename': str(plot_name),
            }
        }
    return config


def set_plot_name(db):
    if db == 'pf':
        plot_name = "Pareto Front"
    elif db == 'obj':
        plot_name = "Objective Data"
    return plot_name


def set_database(moop, db, points):
    if db == 'pf':
        database = pd.DataFrame(moop.getPF())
    elif db == 'obj':
        database = pd.DataFrame(moop.getObjectiveData())
    if moop.getConstraintType() is None:
        df = database
    else:
        if points == 'satisfied':
            constraints = moop.getConstraintType().names
            df = database.copy(deep=True)
            for constraint in constraints:
                indices = df[df[constraint] > 0].index
                df.drop(indices, inplace=True)
                df.reset_index(inplace=True)
        elif points == 'violated':
            constraints = moop.getConstraintType().names
            df = database.copy(deep=True)
            for constraint in constraints:
                indices = df[df[constraint] <= 0].index
                df.drop(indices, inplace=True)
                df.reset_index(inplace=True)
        elif points == 'all':
            df = database
        elif points == 'none':
            df = database[0:0]
    return df


def set_hover_info(database, i):
    hover_info = ""
    for key in database.columns:
        hover_info += str(key)
        hover_info += ": "
        hover_info += str(database[key][i])
        hover_info += "<br>"
        # since plotly is JavaScript-based
        # it uses HTML string formatting
    return hover_info


def check_inputs(
    db,
    output,
    axes,
    points,
    height,
    width,
    font,
    fontsize,
    background_color,
    screenshot,
    image_export_format,
    data_export_format,
    verbose,
    dev_mode,
    pop_up,
    port,
):
    try:
        if (db == 'pf' or
            db == 'obj'):
            pass
        else:
            raise ValueError(str(db) + " is an invalid value for 'db'")
    except:
        raise ValueError(str(db) + " is an invalid value for 'db'")

    try:
        if (output == 'dash' or
            output == 'html' or
            output == 'svg' or
            output == 'pdf' or
            output == 'eps' or
            output == 'jpeg' or
            output == 'png' or
            output == 'webp'):
            pass
        else:
            raise ValueError(str(output) + " is an invalid value for 'output'")
    except:
        raise ValueError(str(output) + " is an invalid value for 'output'")

    try:
        if (axes == 'objectives'):
            pass
        else:
            raise ValueError(str(axes) + " is an invalid value for 'axes'")
    except:
        raise ValueError(str(axes) + " is an invalid value for 'axes'")

    try:
        if (points == 'satisfied' or
            points == 'violated' or
            points == 'all' or
            points == 'none'):
            pass
        else:
            raise ValueError(str(points) + " is an invalid value for 'points'")
    except:
        raise ValueError(str(points) + " is an invalid value for 'points'")

    if (height == 'auto' or
        int(height) >= 1):
        pass
    else:
        raise ValueError(str(height) + " is an invalid value for 'height'")

    try:
        if (width == 'auto' or
            int(width) >= 1):
            pass
        else:
            raise ValueError(str(width) + " is an invalid value for 'width'")
    except:
        raise ValueError(str(width) + " is an invalid value for 'width'")

    try:
        if (str(type(str(font))) == "<class 'str'>"):
            pass
        else:
            raise ValueError(str(font) + " is an invalid value for 'font'")
    except:
        raise ValueError(str(font) + " is an invalid value for 'font'")

    try:
        if (fontsize == 'auto' or
           (int(fontsize) >= 1 and
            int(fontsize) <= 100)):
            pass
        else:
            raise ValueError(str(fontsize) + " is an invalid value for 'fontsize'")
    except:
        raise ValueError(str(fontsize) + " is an invalid value for 'fontsize'")

    try:
        if (str(type(str(background_color))) == "<class 'str'>"):
            pass
        else:
            raise ValueError(str(background_color) + " is an invalid value for 'background_color'")
    except:
        raise ValueError(str(background_color) + " is an invalid value for 'background_color'")

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
    except:
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
    except:
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
    except:
        message = str(data_export_format)
        message += " is an invalid value for 'data_export_format'"
        raise ValueError(message)

    try:
        if (verbose or not verbose):
            pass
        else:
            raise ValueError(str(verbose) + " is an invalid value for 'verbose'")
    except:
        raise ValueError(str(verbose) + " is an invalid value for 'verbose'")

    try:
        if (dev_mode or not dev_mode):
            pass
        else:
            raise ValueError(str(dev_mode) + " is an invalid value for 'dev_mode'")
    except:
        raise ValueError(str(dev_mode) + " is an invalid value for 'dev_mode'")

    try:
        if (pop_up or not pop_up):
            pass
        else:
            raise ValueError(str(pop_up) + " is an invalid value for 'pop_up'")
    except:
        raise ValueError(str(pop_up) + " is an invalid value for 'pop_up'")

    try:
        if (port[0:4] == 'http'):
            pass
        else:
            raise ValueError(str(port) + " is an invalid value for 'port'")
    except:
        raise ValueError(str(port) + " is an invalid value for 'port'")

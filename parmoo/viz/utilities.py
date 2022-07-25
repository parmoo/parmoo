import pandas as pd


def customize(
    fig,
    font,
    fontsize,
    background_color,
    margins,
):
    if font != 'auto':
        fig.update_layout(
            font=dict(
                family=font
            )
        )
    if fontsize != 'auto':
        fig.update_layout(
            font=dict(
                size=int(fontsize)
            )
        )
    if background_color != 'auto':
        pass
    if margins != 'auto':
        pass
    return fig


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
    elif file_type == 'pdf':
        fig.write_image(plot_name + ".pdf")
    elif file_type == 'svg':
        fig.write_image(plot_name + ".svg")
    elif file_type == 'webp':
        fig.write_image(plot_name + ".webp")
    elif file_type == 'jpeg':
        fig.write_image(plot_name + ".jpeg")
    elif file_type == 'png':
        fig.write_image(plot_name + ".png")
    else:
        message = "ParMOO does not support '" + file_type + "'.\n"
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


def configure(height, width, plot_name):

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
    else:
        raise ValueError(str(db) + "is invalid argument for 'db'")
    return plot_name


def set_database(moop, db):
    if db == 'pf':
        database = pd.DataFrame(moop.getPF())
    elif db == 'obj':
        database = pd.DataFrame(moop.getObjectiveData())
    else:
        raise ValueError(str(db) + "is invalid argument for 'db'")
    return database


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

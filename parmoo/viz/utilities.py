import pandas as pd


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


def configure(height, width, plotName):

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

# def output()
#     if output == 'dash':
#         buildDashApp(moop=moop,
#                      db=db,
#                      fig=fig,
#                      config=config,
#                      verbose=verbose,
#                      hot_reload=hot_reload,
#                      pop_up=pop_up,
#                      port=port,)
#     elif output == 'no_dash':
#         fig.show(config=config)
#     else:
#         exportFile(fig=fig,
#                    plotName=plotName,
#                    fileType=output)


def setPlotName(db):
    if db == 'pf':
        plotName = "Pareto Front"
    elif db == 'obj':
        plotName = "Objective Data"
    else:
        raise ValueError(str(db) + "is invalid argument for 'db'")
    return plotName


def setDatabase(moop, db):
    if db == 'pf':
        database = pd.DataFrame(moop.getPF())
    elif db == 'obj':
        database = pd.DataFrame(moop.getObjectiveData())
    else:
        raise ValueError(str(db) + "is invalid argument for 'db'")
    return database


def setHoverInfo(database, i):
    hoverInfo = ""
    for key in database.columns:
        hoverInfo += str(key)
        hoverInfo += ": "
        hoverInfo += str(database[key][i])
        hoverInfo += "<br>"
        # since plotly is JavaScript-based
        # it uses HTML string formatting
    return hoverInfo

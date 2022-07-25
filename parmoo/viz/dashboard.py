import dash
# from dash import dash_table
from dash import html
from dash import dcc
from dash import Input, Output
import pandas as pd
from os import environ
from webbrowser import open_new
from .graph import (generate_scatter,
                    generate_parallel,
                    generate_radar,)
from .utilities import configure, set_plot_name, set_database
from warnings import warn
# import base64

# parmoo_logo = '/Users/hyrumdickinson/parmoo/parmoo/viz/logo-ParMOO.png'
# parmoo_logo_encoded = base64.b64encode(open(parmoo_logo, 'rb').read())


# all dash docs examples build the dash app in an independent script
# by making dash app construction a function dependent
# on calls from a plotting function, we choose a different
# structure than what's used by most apps. This is because
# the purpose of our dash app is not to build an analytics
# dashboard (and put information inside). The purpose is to
# make various kinds of plots and have a consistent functionality
# wrapper around them


def build_dash_app(plot_type,
                   moop,
                   db,
                   height,
                   width,
                   verbose,
                   font,
                   hot_reload,
                   pop_up,
                   port,
                   objectives_only=True,):

    # * define database
    # (initially, all graph data is selected)
    database = set_database(moop, db=db)
    plot_name = set_plot_name(db=db)

    # * create app
    app = dash.Dash(__name__)
    selection_indexes = []

    # * create plot
    graph = generate_graph(plot_type=plot_type,
                           moop=moop,
                           db=db,
                           height=height,
                           width=width,
                           verbose=verbose,
                           font=font,
                           objectives_only=objectives_only,)

    config = configure(height=height,
                       width=width,
                       plot_name=plot_name,)

    # * lay out app
    app.layout = html.Div(children=[
        # * parmoo logo
        # html.Img(
        #     src='data:image/png;base64,{}'.format(parmoo_logo_encoded)
        # ),
        # * main plot
        dcc.Graph(
            id='parmoo_graph',
            figure=graph,
            config=config,
        ),
        dcc.Store(
            id='selection',
        ),
        # * download dataset button
        html.Button(
            children='Download dataset as CSV',
            id='download_dataset_button',
        ),
        dcc.Download(
            id='dataset_download_csv',
        ),
        html.Br(),
        html.Br(),
        # * download selection button
        html.Button(
            children='Download selection as CSV',
            id='download_selection_button',
        ),
        dcc.Download(
            id='selection_download_csv',
        ),
        html.Br(),
        html.Br(),
        dcc.Dropdown(
            ['Open Sans',
             'Times New Roman',
             'Verdana',
             'Arial',
             'Calibri',],
            placeholder="Select a font",
            id='font_selection_downdown'),
    ])

    # * functionality of dataset download button
    @app.callback(
        Output(
            component_id='dataset_download_csv',
            component_property='data'),
        Input(
            component_id='download_dataset_button',
            component_property='n_clicks'),
    )
    def download_dataset(n_clicks):
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate
        else:
            database.index.name = 'index'
            return dict(
                filename=str(plot_name) + ".csv",
                content=database.to_csv(),
            )

    # * create object holding selected data
    @app.callback(
        Output(
            component_id='selection',
            component_property='data'),
        Input(
            component_id='parmoo_graph',
            component_property='selectedData'),
    )
    def store_selection(selectedData):
        if selectedData is None:
            raise dash.exceptions.PreventUpdate
        else:
            pointskey = selectedData['points']
            for index in range(len(pointskey)):
                level1 = pointskey[index]
                point_index = level1['pointIndex']
                selection_indexes.append(point_index)

    # * functionality of selection download button
    @app.callback(
        Output(
            component_id='selection_download_csv',
            component_property='data'),
        Input(
            component_id='download_selection_button',
            component_property='n_clicks'),
    )
    def download_selection(n_clicks):
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate
        else:
            selection_db = database.iloc[:0, :].copy()
            for i in selection_indexes:
                selection_db = pd.concat([selection_db, database.iloc[[i]]])
                selection_db.index.name = 'index'
            selection_db.drop_duplicates(inplace=True)
            selection_db.sort_index(inplace=True)
            return dict(
                filename="selected_data.csv",
                content=selection_db.to_csv(),
            )

    # * functionality of select font button
    @app.callback(
        Output(
            component_id='parmoo_graph',
            component_property='figure'),
        Input(
            component_id='font_selection_downdown',
            component_property='value'),
    )
    def update_font(value):
        if value != 'reset to default':
            font = value
        else:
            font = 'auto'
        graph = generate_graph(plot_type=plot_type,
                               moop=moop,
                               db=db,
                               height=height,
                               width=width,
                               verbose=verbose,
                               font=font,
                               objectives_only=objectives_only,)
        return graph

    # * pop_up
    if pop_up:
        if not environ.get("WERKZEUG_RUN_MAIN"):
            open_new(port)

    # * run application
    if hot_reload:
        app.run(
            debug=True,
            dev_tools_hot_reload=True,
        )
    elif not hot_reload:
        app.run(
            debug=True,
            dev_tools_hot_reload=False,
        )
    else:
        message = str(hot_reload) + " is an invalid value for 'hot_reload'. "
        message += "\nInstead, use on of the boolean values 'True' and 'False'"
        raise ValueError(message)


def generate_graph(plot_type,
                   moop,
                   db,
                   height,
                   width,
                   verbose,
                   font,
                   objectives_only,):
    if plot_type == 'scatter':
        graph = generate_scatter(moop,
                                 db=db,
                                 height=height,
                                 width=width,
                                 verbose=verbose,
                                 font=font,
                                 objectives_only=objectives_only,)
    elif plot_type == 'parallel':
        graph = generate_parallel(moop,
                                  db=db,
                                  height=height,
                                  width=width,
                                  verbose=verbose,
                                  font=font,
                                  objectives_only=objectives_only,)
    elif plot_type == 'radar':
        graph = generate_radar(moop,
                               db=db,
                               height=height,
                               width=width,
                               verbose=verbose,
                               font=font,
                               objectives_only=objectives_only,)
    else:
        warn("invalid plot_type")

    return graph

import dash
# from dash import dash_table
from dash import html
from dash import dcc
from dash import Input, Output
import pandas as pd
from os import environ
from webbrowser import open_new


# all dash docs examples build the dash app in an independent script
# by making dash app construction a function dependent
# on calls from a plotting function, we choose a different
# structure than what's used by most apps. This is because
# the purpose of our dash app is not to build an analytics
# dashboard (and put information inside). The purpose is to
# make various kinds of plots and have a consistent functionality
# wrapper around them


def buildDashApp(moop,
                 db,
                 fig,
                 config,
                 verbose,
                 hot_reload,
                 pop_up,):

    # * define database
    # (initially, all graph data is selected)
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

    # * create app
    app = dash.Dash(__name__)
    selection_indexes = []
    app.layout = html.Div(children=[
        # * header stuff (we don't really need this)
        html.H1(
            id='header',
            children='ParMOO data viz',
        ),
        html.Div(
            id='subheader',
            children='Interact with your MOOP results',
        ),
        # * main plot
        dcc.Graph(
            id='parmoo_plot',
            figure=fig,
            config=config,
        ),
        dcc.Store(
            id='selection',
        ),
        # * download dataset button
        html.Button(
            children='Download dataset as CSV',
            id='dataset_button_text',
        ),
        dcc.Download(
            id='dataset_download_csv',
        ),
        html.Br(),
        html.Br(),
        # * download selection button
        html.Button(
            children='Download selection as CSV',
            id='selection_button_text',
        ),
        dcc.Download(
            id='selection_download_csv',
        ),
    ])

    # * functionality of dataset download button
    @app.callback(
        Output(
            component_id='dataset_download_csv',
            component_property='data'),
        Input(
            component_id='dataset_button_text',
            component_property='n_clicks'),
    )
    def download_dataset(n_clicks):
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate
        else:
            database.index.name = 'index'
            return dict(
                filename=str(plotName) + ".csv",
                content=database.to_csv(),
            )

    # * create object holding selected data
    @app.callback(
        Output(
            component_id='selection',
            component_property='data'),
        Input(
            component_id='parmoo_plot',
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
            component_id='selection_button_text',
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
    if pop_up:
        if not environ.get("WERKZEUG_RUN_MAIN"):
            open_new('http://127.0.0.1:8050/')

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

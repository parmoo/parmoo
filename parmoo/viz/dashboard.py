
import dash
from dash import dash_table
from dash import html
from dash import dcc
from dash import Input, Output
import pandas as pd
import io


# all examples build the dash app in an independent script
# by making dash app construction a function dependent
# on calls from a plotting function, we choose a different
# structure than what's used by most apps. This is because
# the purpose of our dash app is not to build an analytics
# dashboard (and put information inside). The purpose is to
# make various kinds of plots and have a consistent functionality
# wrapper around them
def buildDashApp(moop, db, fig):

    # * define database
    # (initially, all graph data is selected)
    if db == 'pf':
        database = moop.getPF()
    elif db == 'obj':
        database = moop.getObjectiveData()
    else:
        message = "'" + str(db) + "' is not an acceptible value for 'db'\n"
        message += "Consider using 'pf' or 'obj' instead."
        raise ValueError(message)

    # * create app
    app = dash.Dash(__name__)
    app.layout = html.Div(children=[
        # * header stuff (we don't really need this)
        html.H1(
            children='ParMOO data viz',
        ),
        html.Div(
            children='Interact with your MOOP results',
        ),
        # * main plot
        dcc.Graph(
            id='parmoo-plot',
            figure=fig,
        ),
        # * download button
        html.Button(
            children='Download dataset as CSV',
            id='button_text',
        ),
        dcc.Download(
            id='download_csv',
        ),
        # # * csv export data
        # dash_table.DataTable(
        #     data=database,
        #     columns=[{"name": key} for key in database.dtype.names],
        #     id='csv_data',
        #     export_format='csv'
        # )
    ])

    # * functionality of download button
    @app.callback(
        Output(
            component_id='download_csv',
            component_property='data'),
        Input(
            component_id='button_text',
            component_property='n_clicks'),
    )
    def func(n_clicks):
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate
        else:
            pandasData = pd.DataFrame(database).to_csv()
            return dict(
                filename="selected_data.csv",
                content=pandasData,
            )

    app.run(
        debug=True,
        dev_tools_hot_reload=True
    )

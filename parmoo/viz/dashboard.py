
import dash
from dash import html
from dash import dcc
from dash import Input, Output


def buildDashApp(moop, db, fig):

    # define selected data
    # initially, all data in graph is selected
    if db == 'pf':
        selected_data = moop.getPF()
    elif db == 'obj':
        selected_data = moop.getObjectiveData()
    else:
        message = "'" + str(db) + "' is not an acceptible value for 'db'\n"
        message += "Consider using 'pf' or 'obj' instead."
        raise ValueError(message)

    app = dash.Dash(__name__)

    app.layout = html.Div(children=[
        html.H1(children='ParMOO data viz'),

        html.Div(children='''
            Interact with your MOOP results
        '''),

        dcc.Graph(
            id='parmoo-plot',
            figure=fig
        ),
        html.Button('Download CSV', id='button_text'),
        dcc.Download(id='download_csv_index')
    ])

    @app.callback(
        Output(
            component_id='download_csv_index',
            component_property='data'),
        Input(
            component_id='button_text',
            component_property='n_clicks')
    )
    def func(n_clicks):
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate
        else:
            return dict(content="Hello world!", filename="hello.txt")

    app.run(debug=True, dev_tools_hot_reload=True)

#
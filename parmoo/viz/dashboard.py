import pandas as pd
from os import environ
from webbrowser import open_new
from warnings import warn
from dash import (
    Dash,
    callback_context,
    Input,
    Output,
    html,
    dcc,
    exceptions,
)
from .graph import (
    generate_scatter,
    generate_parallel,
    generate_radar,
)
from .utilities import (
    set_plot_name,
    set_database
)
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

class Dash_App:

    def __init__(
        self,
        moop,
        plot_type,
        db,
        axes,
        specificaxes,
        height,
        width,
        font,
        fontsize,
        background_color,
        margins,
        screenshot,
        dummy2,
        dummy3,
        dummy4,
        dummy5,
        dummy6,
        verbose,
        hot_reload,
        pop_up,
        port,
    ):
        # * define independent state
        self.moop = moop
        self.plot_type = plot_type
        self.db = db
        self.axes = axes
        self.specificaxes = specificaxes
        self.height = height
        self.width = width
        self.font = font
        self.fontsize = fontsize
        self.background_color = background_color
        self.margins = margins
        self.screenshot = screenshot
        self.dummy2 = dummy2
        self.dummy3 = dummy3
        self.dummy4 = dummy4
        self.dummy5 = dummy5
        self.dummy6 = dummy6
        self.verbose = verbose
        self.hot_reload = hot_reload
        self.pop_up = pop_up
        self.port = port

        # * define dependent state
        self.selection_indexes = []
        self.plot_name = set_plot_name(db=self.db)
        self.database = set_database(moop, db=self.db)
        self.graph = self.generate_graph()
        self.config = self.configure()

        # * initialize app
        app = Dash(__name__)
        # * lay out app
        app.layout = html.Div(children=[
            # * parmoo logo
            # html.Img(
            #     src='data:image/png;base64,{}'.format(parmoo_logo_encoded)
            # ),
            # * main plot
            dcc.Graph(
                id='parmoo_graph',
                figure=self.graph,
                config=self.config,
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
            dcc.Input(
                placeholder='Select font',
                type='text',
                value='',
                id='font_selection_input',
            ),
            dcc.Input(
                placeholder='Select font size',
                type='number',
                value='',
                id='font_size_input',
            ),
            dcc.Input(
                placeholder='Select graph width',
                type='number',
                value='',
                id='graph_width_input',
            ),
            dcc.Input(
                placeholder='Select graph height',
                type='number',
                value='',
                id='graph_height_input',
            ),
            dcc.Input(
                placeholder='Select margin size',
                type='number',
                value='',
                id='graph_margins_input',
            ),
            dcc.Input(
                placeholder='Select graph title',
                type='text',
                value='',
                id='plot_name_input',
            ),
            dcc.Dropdown(
                options=['svg',
                         'png',
                         'jpeg',
                         'webp'],
                placeholder='Select screenshot format',
                id='screenshot_dropdown',
            ),
            dcc.Dropdown(
                options=['Scatterplot',
                         'Parallel Coordinates plot',
                         'Radar plot'],
                placeholder='Select plot type',
                id='plot_type_dropdown',
            ),
            dcc.Dropdown(
                options=['Pareto Front',
                         'Objective data'],
                placeholder='Select database',
                id='database_dropdown',
            ),
            dcc.Dropdown(
                options=['White',
                         'Grey',
                         'Black',
                         'Red',
                         'Yellow',
                         'Blue',
                         'Green',
                         'Orange',
                         'Purple'],
                placeholder='Select background color',
                id='background_color_dropdown',
            )
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
                raise exceptions.PreventUpdate
            else:
                self.database.index.name = 'index'
                return dict(
                    filename=str(self.plot_name) + ".csv",
                    content=self.database.to_csv(),
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
                raise exceptions.PreventUpdate
            else:
                pointskey = selectedData['points']
                for index in range(len(pointskey)):
                    level1 = pointskey[index]
                    point_index = level1['pointIndex']
                    self.selection_indexes.append(point_index)

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
                raise exceptions.PreventUpdate
            else:
                selection_db = self.database.iloc[:0, :].copy()
                for i in self.selection_indexes:
                    selection_db = pd.concat(
                        [selection_db,
                         self.database.iloc[[i]]]
                    )
                    selection_db.index.name = 'index'
                selection_db.drop_duplicates(inplace=True)
                selection_db.sort_index(inplace=True)
                return dict(
                    filename="selected_data.csv",
                    content=selection_db.to_csv(),
                )

        # * configure graph
        # TODO whenever possible, configure instead of updating or regenerating
        @app.callback(
            Output(
                component_id='parmoo_graph',
                component_property='config',),
            # height
            Input(
                component_id='graph_height_input',
                component_property='value',),
            # width
            Input(
                component_id='graph_width_input',
                component_property='value',),
            # screenshot
            Input(
                component_id='screenshot_dropdown',
                component_property='value',),
            # title
            Input(
                component_id='plot_name_input',
                component_property='value',),
            prevent_initial_call=True
        )
        def update_config(
            height_value,
            width_value,
            screenshot_value,
            plot_name_value,
        ):
            triggered_id = callback_context.triggered[0]['prop_id']
            if 'graph_height_input.value' == triggered_id:
                self.height = height_value
                return self.configure()
            elif 'graph_width_input.value' == triggered_id:
                self.width = width_value
                return self.configure()
            elif 'plot_name_input.value' == triggered_id:
                self.plot_name = plot_name_value
                return self.configure()
            elif 'screenshot_dropdown.value' == triggered_id:
                self.screenshot = screenshot_value
                return self.configure()

        # * regenerate or update graph
        # TODO whenever possible, update instead of regenerating
        @app.callback(
            Output(
                component_id='parmoo_graph',
                component_property='figure',),
            # font - update
            Input(
                component_id='font_selection_input',
                component_property='value',),
            # font size - update
            Input(
                component_id='font_size_input',
                component_property='value',),
            # margins
            Input(
                component_id='graph_margins_input',
                component_property='value',),
            # background color - update
            Input(
                component_id='background_color_dropdown',
                component_property='value',),
            # plot type - regenerate
            Input(
                component_id='plot_type_dropdown',
                component_property='value',),
            # database - regenerate
            Input(
                component_id='database_dropdown',
                component_property='value',),
            prevent_initial_call=True
        )
        def update_graph(
            font_value,
            size_value,
            margins_value,
            background_color_value,
            plot_type_value,
            database_value,
        ):
            triggered_id = callback_context.triggered[0]['prop_id']
            if 'font_selection_input.value' == triggered_id:
                if font_value != "":
                    self.font = font_value
                    return update_font()
            elif 'font_size_input.value' == triggered_id:
                self.fontsize = size_value
                return update_font_size()
            elif 'graph_margins_input.value' == triggered_id:
                self.margins = margins_value
                return update_margins()
            elif 'background_color_dropdown.value' == triggered_id:
                self.background_color = background_color_value
                return update_background_color()
            elif 'plot_type_dropdown.value' == triggered_id:
                if plot_type_value == 'Scatterplot':
                    self.plot_type = 'scatter'
                elif plot_type_value == 'Parallel Coordinates plot':
                    self.plot_type = 'parallel'
                elif plot_type_value == 'Radar plot':
                    self.plot_type = 'radar'
                return update_plot_type()
            elif 'database_dropdown.value' == triggered_id:
                self.database = database_value
                return update_database()

        # * functionality of select font input
        def update_font():
            self.graph.update_layout(
                font=dict(
                    family=self.font
                )
            )
            return self.graph

        # * functionality of select font size input
        def update_font_size():
            self.graph.update_layout(
                font=dict(
                    size=int(self.fontsize)
                )
            )
            return self.graph

        # * functionality of graph margins input
        def update_margins():
            pass

        # * functionality of background color dropdown
        def update_background_color():
            pass

        # * functionality of plot type dropdown
        def update_plot_type():
            return self.generate_graph()

        # * functionality of database dropdown
        def update_database():
            return self.generate_graph()

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
            message = str(hot_reload) + " is an invalid value for 'hot_reload'"
            message += "\n'hot_reload' accepts boolean values only"
            raise ValueError(message)

    def generate_graph(self):
        if self.plot_type == 'scatter':
            self.graph = generate_scatter(
                moop=self.moop,
                db=self.db,
                axes=self.axes,
                specificaxes=self.specificaxes,
                height=self.height,
                width=self.width,
                font=self.font,
                fontsize=self.fontsize,
                background_color=self.background_color,
                margins=self.margins,
                screenshot=self.screenshot,
                dummy2=self.dummy2,
                dummy3=self.dummy3,
                dummy4=self.dummy4,
                dummy5=self.dummy5,
                dummy6=self.dummy6,
                verbose=self.verbose,
            )
        elif self.plot_type == 'parallel':
            self.graph = generate_parallel(
                moop=self.moop,
                db=self.db,
                axes=self.axes,
                specificaxes=self.specificaxes,
                height=self.height,
                width=self.width,
                font=self.font,
                fontsize=self.fontsize,
                background_color=self.background_color,
                margins=self.margins,
                screenshot=self.screenshot,
                dummy2=self.dummy2,
                dummy3=self.dummy3,
                dummy4=self.dummy4,
                dummy5=self.dummy5,
                dummy6=self.dummy6,
                verbose=self.verbose,
            )
        elif self.plot_type == 'radar':
            self.graph = generate_radar(
                moop=self.moop,
                db=self.db,
                axes=self.axes,
                specificaxes=self.specificaxes,
                height=self.height,
                width=self.width,
                font=self.font,
                fontsize=self.fontsize,
                background_color=self.background_color,
                margins=self.margins,
                screenshot=self.screenshot,
                dummy2=self.dummy2,
                dummy3=self.dummy3,
                dummy4=self.dummy4,
                dummy5=self.dummy5,
                dummy6=self.dummy6,
                verbose=self.verbose,
            )
        else:
            warn("invalid plot_type")

        return self.graph

    def configure(self):
        if self.height != 'auto' and self.width != 'auto':
            self.config = {
                'displaylogo': False,
                'displayModeBar': True,
                'toImageButtonOptions': {
                    'format': self.screenshot,
                    'filename': str(self.plot_name),
                    'height': int(self.height),
                    'width': int(self.width),
                    'scale': 1
                }
            }
        else:
            self.config = {
                'displaylogo': False,
                'displayModeBar': True,
                'toImageButtonOptions': {
                    'format': self.screenshot,
                    'filename': str(self.plot_name),
                }
            }

        return self.config

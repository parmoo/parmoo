import pandas as pd
import plotly.io as pio
from os import environ
from webbrowser import open_new
from warnings import warn
import logging
from dash import (
    Dash,
    Input,
    Output,
    html,
    dcc,
    exceptions,
    callback_context,
    no_update,
)
from .graph import (
    generate_scatter,
    generate_parallel,
    generate_radar,
)
from .utilities import (
    set_plot_name,
    set_database,
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
        logging.info('initializing dashboard')

        # ! STATE

        # * define independent state
        self.moop = moop
        self.plot_type = plot_type
        self.db = db
        self.points = points
        self.height = height
        self.width = width
        self.font = font
        self.fontsize = fontsize
        self.background_color = background_color
        self.screenshot = screenshot
        self.image_export_format = image_export_format
        self.data_export_format = data_export_format
        self.dev_mode = dev_mode
        self.pop_up = pop_up
        self.port = port
        # * define dependent state
        self.selection_indexes = []
        self.constraint_range = self.set_constraint_range(None)
        self.plot_name = set_plot_name(db=self.db)
        self.database = set_database(moop, db=self.db, points=self.points)
        self.graph = self.generate_graph()
        self.config = self.configure()

    # ! LAYOUT

        # * initialize app
        app = Dash(__name__)
        # * lay out app
        app.layout = html.Div(children=[
            # * parmoo logo
            # html.Img(
            #     src='data:image/png;base64,{}'.format(parmoo_logo_encoded)
            # ),
            dcc.Dropdown(
                id='plot_type_dropdown',
                options=['Scatterplot',
                         'Parallel Coordinates plot',
                         'Radar plot'],
                placeholder='Change plot type',
            ),
            dcc.Dropdown(
                id='database_dropdown',
                options=[
                    'Pareto Front',
                    'Objective Data'
                ],
                placeholder='Change dataset',
            ),
            html.Button(
                children='Export all data',
                id='download_dataset_button',
            ),
            dcc.Download(
                id='download_dataset_dcc',
            ),
            html.Button(
                children='Export selected data',
                id='download_selection_button',
            ),
            dcc.Download(
                id='download_selection_dcc',
            ),
            html.Button(
                children='Export image to working directory',
                id='download_image_button',
            ),
            dcc.Store(
                id='image'
            ),
            dcc.Checklist(
                id='constraint_checkboxes',
                options=[
                    {'label': 'show constraint-satisfying points',
                     'value': 'constraint_satisfying'},
                    {'label': 'show constraint-violating points',
                     'value': 'constraint_violating'},
                ],
                value=['constraint_satisfying'],
                inline=True,
            ),
            html.Br(),
            # * main plot
            dcc.Graph(
                id='parmoo_graph',
                figure=self.graph,
                config=self.config,
            ),
            html.Br(),
            html.Button(
                children='Show graph customization options',
                id='show_customization_options',
            ),
            html.Button(
                children='Hide graph customization options',
                id='hide_customization_options',
                style=dict(display='none'),
            ),
            html.Button(
                children='Show export options',
                id='show_export_options',
            ),
            html.Button(
                children='Hide export options',
                id='hide_export_options',
                style=dict(display='none'),
            ),
            dcc.Store(
                id='selection',
            ),
            dcc.Input(
                id='font_selection_input',
                placeholder='Set font',
                type='text',
                value='',
                debounce=True,
                style=dict(display='none'),
            ),
            dcc.Input(
                id='font_size_input',
                placeholder='Set font size',
                type='number',
                value='',
                min=1,
                max=100,
                debounce=True,
                style=dict(display='none'),
            ),
            dcc.Input(
                id='graph_width_input',
                placeholder='Set graph width',
                type='number',
                value='',
                min=10,
                step=1,
                style=dict(display='none'),
            ),
            dcc.Input(
                id='graph_height_input',
                placeholder='Set graph height',
                type='number',
                value='',
                min=10,
                step=1,
                style=dict(display='none'),
            ),
            dcc.Input(
                id='plot_name_input',
                placeholder='Set plot name',
                type='text',
                value='',
                debounce=True,
                style=dict(display='none'),
            ),
            dcc.Dropdown(
                id='background_color_dropdown',
                options=[
                    'White',
                    'Grey',
                    'Black',
                    'Transparent',
                ],
                placeholder='Set background color',
                style=dict(display='none'),
            ),
            dcc.Dropdown(
                id='image_export_format_dropdown',
                options=[
                    'PNG',
                    'JPEG',
                    'PDF',
                    'SVG',
                    'EPS',
                    'HTML',
                    'WebP'
                ],
                placeholder='Set image export format',
                style=dict(display='none'),
            ),
            dcc.Store(
                id='image_export_format_store',
                storage_type='local',
            ),
            dcc.Dropdown(
                id='data_export_format_dropdown',
                options=[
                    'CSV',
                    'JSON',
                ],
                placeholder='Set data export format',
                style=dict(display='none'),
            ),
            dcc.Store(
                id='data_export_format_store',
                storage_type='local',
            )
        ])

    # ! CALLBACKS

        # * show customization options
        @app.callback(
            Output(
                component_id='show_customization_options',
                component_property='style',),
            Output(
                component_id='hide_customization_options',
                component_property='style',),
            Output(
                component_id='font_selection_input',
                component_property='style',),
            Output(
                component_id='font_size_input',
                component_property='style',),
            Output(
                component_id='graph_width_input',
                component_property='style',),
            Output(
                component_id='graph_height_input',
                component_property='style',),
            Output(
                component_id='plot_name_input',
                component_property='style',),
            Output(
                component_id='background_color_dropdown',
                component_property='style',),
            Input(
                component_id='show_customization_options',
                component_property='n_clicks',),
            Input(
                component_id='hide_customization_options',
                component_property='n_clicks',),
            prevent_initial_call=True
        )
        def update_customization_components(
            s_clicks,
            h_clicks,
        ):
            triggered_id = callback_context.triggered[0]['prop_id']
            if triggered_id == 'show_customization_options.n_clicks':
                logging.info("'show_customization_options.n_clicks' triggered")
                return self.evaluate_customization_options('show', s_clicks)
            elif triggered_id == 'hide_customization_options.n_clicks':
                logging.info("'hide_customization_options.n_clicks' triggered")
                return self.evaluate_customization_options('hide', h_clicks)

        # * show export options
        @app.callback(
            Output(
                component_id='show_export_options',
                component_property='style',),
            Output(
                component_id='hide_export_options',
                component_property='style',),
            Output(
                component_id='image_export_format_dropdown',
                component_property='style',),
            Output(
                component_id='data_export_format_dropdown',
                component_property='style',),
            Input(
                component_id='show_export_options',
                component_property='n_clicks',),
            Input(
                component_id='hide_export_options',
                component_property='n_clicks',),
            prevent_initial_call=True
        )
        def update_export_components(
            s_clicks,
            h_clicks,
        ):
            triggered_id = callback_context.triggered[0]['prop_id']
            if triggered_id == 'show_export_options.n_clicks':
                logging.info("'show_export_options.n_clicks' triggered")
                return self.evaluate_export_options('show', s_clicks)
            elif triggered_id == 'hide_export_options.n_clicks':
                logging.info("'hide_export_options.n_clicks' triggered")
                return self.evaluate_export_options('hide', h_clicks)

        # * regenerate or update graph
        # TODO whenever possible, update instead of regenerating
        @app.callback(
            Output(
                component_id='parmoo_graph',
                component_property='figure',),
            # height - update
            Input(
                component_id='graph_height_input',
                component_property='value',),
            # width - update
            Input(
                component_id='graph_width_input',
                component_property='value',),
            # font - update
            Input(
                component_id='font_selection_input',
                component_property='value',),
            # font size - update
            Input(
                component_id='font_size_input',
                component_property='value',),
            # background color - update
            Input(
                component_id='background_color_dropdown',
                component_property='value',),
            # plot name - update
            Input(
                component_id='plot_name_input',
                component_property='value',),
            # plot type - regenerate
            Input(
                component_id='plot_type_dropdown',
                component_property='value',),
            # database - regenerate
            Input(
                component_id='database_dropdown',
                component_property='value',),
            # constraint showr - regenerate
            Input(
                component_id='constraint_checkboxes',
                component_property='value'),
            prevent_initial_call=True
        )
        def update_graph(
            height_value,
            width_value,
            font_value,
            font_size_value,
            background_color_value,
            plot_name_value,
            plot_type_value,
            database_value,
            constraint_showr_value,
        ):
            triggered_id = callback_context.triggered[0]['prop_id']
            if 'graph_height_input.value' == triggered_id:
                logging.info("'graph_height_input.value' triggered")
                return self.evaluate_height(height_value)
            elif 'graph_width_input.value' == triggered_id:
                logging.info("'graph_width_input.value' triggered")
                return self.evaluate_width(width_value)
            elif 'font_selection_input.value' == triggered_id:
                logging.info("'font_selection_input.value' triggered")
                return self.evaluate_font(font_value)
            elif 'font_size_input.value' == triggered_id:
                logging.info("'font_size_input.value' triggered")
                return self.evaluate_font_size(font_size_value)
            elif 'background_color_dropdown.value' == triggered_id:
                logging.info("'background_color_dropdown.value' triggered")
                return self.evaluate_background_color(background_color_value)
            elif 'plot_name_input.value' == triggered_id:
                logging.info("'plot_name_input.value' triggered")
                return self.evaluate_plot_name(plot_name_value)
            elif 'plot_type_dropdown.value' == triggered_id:
                logging.info("'plot_type_dropdown.value' triggered")
                return self.evaluate_plot_type(plot_type_value)
            elif 'database_dropdown.value' == triggered_id:
                logging.info("'database_dropdown.value' triggered")
                return self.evaluate_database(database_value)
            elif 'constraint_checkboxes.value' == triggered_id:
                logging.info("'constraint_checkboxes.value' triggered")
                return self.evaluate_constraint_showr(constraint_showr_value)

        # * download dataset
        @app.callback(
            Output(
                component_id='download_dataset_dcc',
                component_property='data'),
            Input(
                component_id='download_dataset_button',
                component_property='n_clicks'),
        )
        def download_dataset(n_clicks):
            logging.info("'download_dataset_button.n_clicks' triggered")
            return self.evaluate_dataset_download(n_clicks)

        # * create object holding selected data
        @app.callback(
            Output(
                component_id='selection',
                component_property='data'),
            Input(
                component_id='parmoo_graph',
                component_property='selectedData'),
            Input(
                component_id='parmoo_graph',
                component_property='restyleData'),
        )
        def store_selection(
            selectedData,
            restyleData,
        ):
            triggered_id = callback_context.triggered[0]['prop_id']
            if 'parmoo_graph.selectedData' == triggered_id:
                logging.info("'parmoo_graph.selectedData' triggered")
                self.evaluate_selected_data(selectedData, 'selectedData')
            elif 'parmoo_graph.restyleData' == triggered_id:
                logging.info("'parmoo_graph.restyleData' triggered")
                if self.plot_type == 'parallel':
                    self.evaluate_selected_data(restyleData, 'restyleData')

        # * download selection
        @app.callback(
            Output(
                component_id='download_selection_dcc',
                component_property='data'),
            Input(
                component_id='download_selection_button',
                component_property='n_clicks'),
        )
        def download_selection(n_clicks):
            logging.info("'download_selection_button.n_clicks' triggered")
            return self.evaluate_selection_download(n_clicks)

        # * update image export format
        @app.callback(
            Output(
                component_id='image_export_format_store',
                component_property='data'),
            Input(
                component_id='image_export_format_dropdown',
                component_property='value'),
        )
        def update_image_export_format(value):
            logging.info("'image_export_format_dropdown.value' triggered")
            self.evaluate_image_export_format(value)

        # * update data export format
        @app.callback(
            Output(
                component_id='data_export_format_store',
                component_property='data'),
            Input(
                component_id='data_export_format_dropdown',
                component_property='value'),
        )
        def update_data_export_format(value):
            logging.info("'data_export_format_dropdown.value' triggered")
            self.evaluate_data_export_format(value)

        # * export image
        @app.callback(
            Output(
                component_id='image',
                component_property='data'),
            Input(
                component_id='download_image_button',
                component_property='n_clicks'),
        )
        def download_image(n_clicks):
            logging.info("'download_image_button.n_clicks' triggered")
            return self.evaluate_image_download(n_clicks)

    # ! EXECUTION

        logging.info('initialized dashboard')

        # * pop_up
        if pop_up:
            if not environ.get("WERKZEUG_RUN_MAIN"):
                open_new(port)

        # * run application
        logging.info('opening dashboard in browser. this might take a while')
        if dev_mode:
            app.run(
                debug=True,
                dev_tools_hot_reload=True,
            )
        elif not dev_mode:
            app.run(
                debug=False,
                dev_tools_hot_reload=False,
            )

    # ! INITIALIZATION HELPERS

    def generate_graph(self):
        if self.plot_type == 'scatter':
            self.graph = generate_scatter(
                moop=self.moop,
                db=self.db,
                points=self.points,
            )
        elif self.plot_type == 'parallel':
            self.graph = generate_parallel(
                moop=self.moop,
                db=self.db,
                points=self.points,
            )
        elif self.plot_type == 'radar':
            self.graph = generate_radar(
                moop=self.moop,
                db=self.db,
                points=self.points,
            )
        else:
            warn("invalid plot_type")

        self.graph = self.update_height()
        self.graph = self.update_width()
        self.graph = self.update_font()
        self.graph = self.update_font_size()
        self.graph = self.update_plot_name()
        self.graph = self.update_background_color()

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

    # ! UPDATE HELPERS

    # * functionality of select height input
    def update_height(self):
        if self.height != 'auto':
            self.graph.update_layout(
                height=int(self.height))
        return self.graph

    # * functionality of select width input
    def update_width(self):
        if self.width != 'auto':
            self.graph.update_layout(
                width=int(self.width))
        return self.graph

    # * functionality of select font input
    def update_font(self):
        if self.font != 'auto':
            self.graph.update_layout(
                font=dict(
                    family=self.font
                )
            )
        return self.graph

    # * functionality of select font size input
    def update_font_size(self):
        if self.fontsize != 'auto':
            self.graph.update_layout(
                font=dict(
                    size=int(self.fontsize)
                )
            )
        return self.graph

    # * functionality of plot name input
    def update_plot_name(self):
        self.graph.update_layout(
            title_text=self.plot_name
        )
        return self.graph

    # * functionality of background color dropdown
    def update_background_color(self):
        if self.background_color != 'auto':
            if self.plot_type == 'scatter':
                self.graph.update_layout(
                    plot_bgcolor=self.background_color,
                    paper_bgcolor=self.background_color,
                )
            elif self.plot_type == 'parallel':
                self.graph.update_layout(
                    paper_bgcolor=self.background_color,
                )
            elif self.plot_type == 'radar':
                self.graph.update_polars(
                    bgcolor=self.background_color,
                )
                self.graph.update_layout(
                    paper_bgcolor=self.background_color,
                )
        return self.graph

    # * functionality of plot type dropdown
    def update_plot_type(self):
        return self.generate_graph()

    # * functionality of database dropdown
    def update_database(self):
        self.database = set_database(
            moop=self.moop,
            db=self.db,
            points=self.points
        )
        if self.plot_name == 'Pareto Front':
            self.plot_name = set_plot_name(db=self.db)
        elif self.plot_name == 'Objective Data':
            self.plot_name = set_plot_name(db=self.db)
        return self.generate_graph()

    # ! CALLBACK HELPERS

    def evaluate_height(self, height_value):
        if height_value is not None:
            self.height = height_value
            return self.update_height()
        else:
            return self.graph

    def evaluate_width(self, width_value):
        if width_value is not None:
            self.width = width_value
            return self.update_width()
        else:
            return self.graph

    def evaluate_font(self, font_value):
        if font_value != '':
            self.font = font_value
            self.graph = self.update_font()
            return self.graph
        else:
            return self.graph

    def evaluate_font_size(self, font_size_value):
        self.fontsize = font_size_value
        self.graph = self.update_font_size()
        return self.graph

    def evaluate_background_color(self, background_color_value):
        if background_color_value == 'Transparent':
            self.background_color = 'rgb(0,0,0,0)'
        else:
            self.background_color = background_color_value
        return self.update_background_color()

    def evaluate_plot_name(self, plot_name_value):
        if plot_name_value != '':
            self.plot_name = plot_name_value
            self.graph = self.update_plot_name()
            return self.graph
        else:
            return self.graph

    def evaluate_plot_type(self, plot_type_value):
        if plot_type_value == 'Scatterplot':
            self.plot_type = 'scatter'
        elif plot_type_value == 'Parallel Coordinates plot':
            self.plot_type = 'parallel'
        elif plot_type_value == 'Radar plot':
            self.plot_type = 'radar'
        return self.update_plot_type()

    def evaluate_database(self, database_value):
        if database_value == 'Pareto Front':
            self.db = 'pf'
        elif database_value == 'Objective Data':
            self.db = 'obj'
        return self.update_database()

    def evaluate_dataset_download(self, n_clicks):
        if n_clicks is None:
            raise exceptions.PreventUpdate
        else:
            self.database.index.name = 'index'
            if self.data_export_format == 'CSV':
                return dict(
                    filename=str(self.plot_name) + ".csv",
                    content=self.database.to_csv(),
                )
            else:
                return dict(
                    filename=str(self.plot_name) + ".json",
                    content=self.database.to_json(),
                )

    def evaluate_selected_data(self, data, type):
        if data is None:
            raise exceptions.PreventUpdate
        else:
            self.selection_indexes = []
            if type == 'selectedData':
                selectedData = data
                pointskey = selectedData['points']
                for index in range(len(pointskey)):
                    level1 = pointskey[index]
                    point_index = level1['pointIndex']
                    self.selection_indexes.append(point_index)
            elif type == 'restyleData':
                self.update_constraint_range(data)
                objectives = self.moop.getObjectiveType().names
                for i, row in self.database.iterrows():
                    row_selected = True
                    for objective in objectives:
                        if row_selected:
                            row_obj_value = row[objective]
                            location = objectives.index(objective)
                            entry_dict = self.constraint_range[location]
                            ranges = entry_dict[list(entry_dict.keys())[0]]
                            if ranges is not None:
                                if row_selected:
                                    for range in ranges:
                                        try:
                                            for rang in range:
                                                for ran in rang:
                                                    pass
                                            row_selected_yet = False
                                            for rang in range:
                                                if not row_selected_yet:
                                                    if (row_obj_value >=
                                                       rang[0] and
                                                       row_obj_value <=
                                                       rang[1]):
                                                        row_selected_yet = True
                                                    else:
                                                        row_selected_yet =False
                                            row_selected = row_selected_yet
                                        except:
                                            if (row_obj_value >= range[0] and
                                               row_obj_value <= range[1]):
                                                row_selected = True
                                            else:
                                                row_selected = False
                    if row_selected:
                        self.selection_indexes.append(i)

    def set_constraint_range(self, restyleData):
        if restyleData is None:
            objectives = self.moop.getObjectiveType().names
            self.constraint_range = [None] * len(objectives)
            count = 0
            for objective in objectives:
                self.constraint_range[count] = dict({objective: None})
                count += 1
            return self.constraint_range
        else:
            return self.update_constraint_range(self, restyleData)

    def update_constraint_range(self, restyleData):
        key_list = restyleData[0]
        for key in key_list:
            location = int(key[11])
            entry_dict = self.constraint_range[location]
            entry_dict[list(entry_dict.keys())[0]] = key_list[key]
        return self.constraint_range

    def evaluate_selection_download(self, n_clicks):
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
            if self.data_export_format == 'CSV':
                return dict(
                    filename="selected_data.csv",
                    content=selection_db.to_csv(),
                )
            else:
                return dict(
                    filename="selected_data.json",
                    content=selection_db.to_json(),
                )

    def evaluate_image_export_format(self, image_export_format_value):
        if image_export_format_value is not None:
            self.image_export_format = image_export_format_value

    def evaluate_data_export_format(self, data_export_format_value):
        if data_export_format_value is not None:
            self.data_export_format = data_export_format_value

    def evaluate_image_download(self, n_clicks):
        if n_clicks is None:
            raise exceptions.PreventUpdate
        else:
            file_name = str(self.plot_name)
            file_name += '.' + str(self.image_export_format)
            if (str(self.image_export_format) == 'HTML' or
               str(self.image_export_format) == 'html'):
                con_tent = pio.write_html(
                        fig=self.graph,
                        file=str(file_name)
                )
            else:
                con_tent = pio.write_image(
                        fig=self.graph,
                        file=str(file_name)
                )

            return dict(
                filename=file_name,
                content=con_tent,
            )

    def evaluate_customization_options(self, action, n_clicks):
        if n_clicks is None:
            raise exceptions.PreventUpdate
        else:
            showr = dict()
            hider = dict(display='none')
            if action == 'show':
                return hider, showr, showr, showr, showr, showr, showr, showr
            else:
                return showr, hider, hider, hider, hider, hider, hider, hider

    def evaluate_export_options(self, action, n_clicks):
        if n_clicks is None:
            return no_update, no_update
        else:
            showr = dict()
            hider = dict(display='none')
            if action == 'show':
                return hider, showr, showr, showr
            else:
                return showr, hider, hider, hider

    def evaluate_constraint_showr(self, value):
        if value is None:
            raise exceptions.PreventUpdate
        else:
            if value == ['constraint_satisfying']:
                self.points = 'constraint_satisfying'
            elif value == ['constraint_violating']:
                self.points = 'constraint_violating'
            elif value == ['constraint_satisfying', 'constraint_violating']:
                self.points = 'all'
            elif value == ['constraint_violating', 'constraint_satisfying']:
                self.points = 'all'
            else:
                self.points = 'none'
        self.database = set_database(
            moop=self.moop,
            db=self.db,
            points=self.points,
        )
        return self.generate_graph()

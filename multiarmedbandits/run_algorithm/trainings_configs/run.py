import dash
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html
from dash.dependencies import ALL, Input, Output, State
from dash.exceptions import PreventUpdate

from multiarmedbandits.environments.common import ArmDistTypes
from multiarmedbandits.run_algorithm.metrics import Algorithms, MetricNames

app = dash.Dash(__name__, prevent_initial_callbacks=True, suppress_callback_exceptions=True)

# metrics = [i for i in MetricNames]
algorithms = [i for i in Algorithms]
all_options = {
    "EpsilonGreedy": ["epsilon", "andererWert"],
    "ExploreThenCommit": ["wert", "wert", "wert"],
    "UCBAlpha": [],
    "LectureUCB": [],
    "BoltzmannSimple": [],
    "BoltzmannGeneral": [],
    "GradientBandit": [],
}


yaml_list = []
number_of_arms = 2


def generate_inputs(config):
    return html.Div(
        [str(config), dcc.Input(id={"type": "dynamic-textbox", "index": config}, placeholder=str(config), type="text")]
    )


def generate_arm_configs(config):
    return html.Div(
        [
            f"Arm {config} mean",
            dcc.Input(id={"type": "dynamic-textbox", "index": f"arm{config}_mean"}, placeholder=str(config), type="text"),
            f"Arm {config} scale",
            dcc.Input(id={"type": "dynamic-textbox", "index": f"arm{config}_scale"}, placeholder=str(config), type="text"),
        ]
    )


app.layout = html.Div(
    [
        html.Div(["Number of arms: ", dcc.Input(id="no_arms", value=2, type="text")]),
        html.Div(id="display-number-of-arms"),
        html.Br(),
        dcc.Dropdown(id="mab_env-dropdown", options=[{"label": k, "value": k} for k in ArmDistTypes], value="bernoulli"),
        html.Hr(),
        html.Div(id="display-inputs", children=[generate_arm_configs(i) for i in range(1, 2)]),
        html.Hr(),
        html.Br(),
        html.Br(),
        dcc.Dropdown(
            id="metrics-dropdown", options=[{"label": k, "value": k} for k in MetricNames], value="regret", multi=True
        ),
        html.Div(id="display-metrics"),
        html.Br(),
        dcc.Dropdown(id="algo-dropdown", options=[{"label": k, "value": k} for k in Algorithms], value="EpsilonGreedy"),
        dbc.Button(
            id="btn-scrape",
            children=["Add to yaml  ", html.I(className="fa fa-download mr-1")],
            color="primary",
            className="mt-1",
        ),
        html.Hr(),
        html.Div(id="configure-algo", children=[generate_inputs(i) for i in all_options["EpsilonGreedy"]]),
        html.Hr(),
        dbc.Col(
            [
                dcc.Loading(
                    id="loading-1",
                    type="circle",
                    children=[
                        dash_table.DataTable(
                            id="datatable-paging",
                            data=[{}],
                            editable=True,
                            style_cell={"whiteSpace": "normal", "lineHeight": "auto", "height": "auto", "fontSize": 12},
                            page_current=0,
                            page_size=10,
                            page_action="native",
                        ),
                        dcc.Download(id="download-table"),
                    ],
                ),
            ]
        ),
    ]
)

@app.callback(Output("display-number-of-arms", "children"), Input("no_arms", "value"))
def update_output(value):
    global number_of_arms
    number_of_arms = value
    print(number_of_arms)
    return f"You have selected {value}"

@app.callback(Output("display-metrics", "children"), Input("metrics-dropdown", "value"))
def update_output(value):
    return f"You have selected {value}"

@app.callback(
    Output("display-inputs", "children"),
    Input("no_arms", "value"),
)
def set_configs_children_number_arms(value):
    return dbc.Col(children=[generate_arm_configs(i) for i in range(1, int(value) + 1)], className="placeholder")


@app.callback(
    Output("configure-algo", "children"),
    Input("algo-dropdown", "value"),
)
def set_configs_children(scraping_setting):
    return dbc.Col(children=[generate_inputs(i) for i in all_options[scraping_setting]], className="placeholder")


"""
App callback: Scraping data from eurlex using the XmlScraper Class
"""


@app.callback(
    Output("datatable-paging", "data"),
    [Input("btn-scrape", "n_clicks")],
    State({"type": "dynamic-textbox", "index": ALL}, "value"),
)
def update_output(n_clicks, value):
    if n_clicks is None:
        raise PreventUpdate
    else:
        print(value)


if __name__ == "__main__":
    app.run_server(debug=True)

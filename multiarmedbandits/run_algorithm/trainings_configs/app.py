from dash import Dash, dcc, html, Input, Output, callback

app = Dash(__name__)

app.layout = html.Div([
    html.H6("Change the value in the text box to see callbacks in action!"),
    html.Div([
        "Input: ",
        dcc.Input(id='my-input', value='initial value', type='text')
    ]),
    
    
    html.Div(id='my-output'),
    html.Br(),
    html.Label('Select what Algorithm you want to test:'),
    dcc.Dropdown(['EpsilonGreedy', 'ExploreThenCommit', 'UCBAlpha', 'LectureUCB',
                'BoltzmannSimple', 'BoltzmannGeneral', 'GradientBandit'], '1', id='demo-dropdown'),
        html.Div(id='dd-output-container'),

])


@callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    return f'Output: {input_value}'


@callback(
    Output('dd-output-container', 'children'),
    Input('demo-dropdown', 'value')
)
def update_output_dropdown(value):
    return f'You have selected {value}'


if __name__ == '__main__':
    app.run(debug=True)






@dataclass
class TestYaml:
    algo: ALgorithms
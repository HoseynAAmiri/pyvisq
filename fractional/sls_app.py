import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np

# Your custom SLS class
from sls_fractional import FractionalSLS

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Slider(min=0.1, max=1, step=0.01, value=1, id='A-slider'),
            dcc.Input(type='number', value=1, step=0.01,
                      id='A-input', style={'width': '100%'}),
            html.P('A')
        ]),
        dbc.Col([
            dcc.Slider(min=0, max=0.99, step=0.01, value=0.35, id='B-slider'),
            dcc.Input(type='number', value=0.35, step=0.01,
                      id='B-input', style={'width': '100%'}),
            html.P('B')
        ]),
        dbc.Col([
            dcc.Slider(min=1, max=50, step=0.1, value=4.0, id='E1-slider'),
            dcc.Input(type='number', value=4.0, step=0.1,
                      id='E1-input', style={'width': '100%'}),
            html.P('E1')
        ]),
        dbc.Col([
            dcc.Slider(min=1, max=50, step=0.1, value=4.0, id='E2-slider'),
            dcc.Input(type='number', value=4.0, step=0.1,
                      id='E2-input', style={'width': '100%'}),
            html.P('E2')
        ]),
        dbc.Col([
            dcc.Slider(min=0.1, max=50, step=1, value=10, id='T1-slider'),
            dcc.Input(type='number', value=10, step=0.1,
                      id='T1-input', style={'width': '100%'}),
            html.P('T1')
        ]),
        dbc.Col([
            dcc.Slider(min=0.1, max=50, step=0.1, value=1, id='T2-slider'),
            dcc.Input(type='number', value=1, step=0.1,
                      id='T2-input', style={'width': '100%'}),
            html.P('T2')
        ]),
        dbc.Col([
            dcc.Slider(min=1, max=10, step=0.1, value=2.0, id='EK-slider'),
            dcc.Input(type='number', value=2.0, step=0.1,
                      id='EK-input', style={'width': '100%'}),
            html.P('EK')
        ])
    ]),
    dcc.Graph(id='stress-strain-plot')
])


@app.callback(
    [Output(f'{param}-slider', 'value') for param in ['A', 'B', 'E1', 'E2', 'T1', 'T2', 'EK']] +
    [Output(f'{param}-input', 'value')
     for param in ['A', 'B', 'E1', 'E2', 'T1', 'T2', 'EK']],
    [Input(f'{param}-slider', 'value') for param in ['A', 'B', 'E1', 'E2', 'T1', 'T2', 'EK']] +
    [Input(f'{param}-input', 'value')
     for param in ['A', 'B', 'E1', 'E2', 'T1', 'T2', 'EK']]
)
def sync_sliders_and_inputs(*vals):
    ctx = dash.callback_context
    if not ctx.triggered:
        # This block executes at initialization: all sliders and inputs set to their initial values
        return vals
    else:
        # Determine which control triggered the callback
        input_id = ctx.triggered[0]['prop_id']
        param, control_type = input_id.split('.')[0].rsplit('-', 1)
        updated_value = ctx.triggered[0]['value']

        # Determine indices for outputs
        index = ['A', 'B', 'E1', 'E2', 'T1', 'T2', 'EK'].index(param)
        return [updated_value if i == index or i == index + 7 else vals[i] for i in range(14)]


@app.callback(
    Output('stress-strain-plot', 'figure'),
    [
        Input('A-slider', 'value'),
        Input('B-slider', 'value'),
        Input('E1-slider', 'value'),
        Input('E2-slider', 'value'),
        Input('T1-slider', 'value'),
        Input('T2-slider', 'value'),
        Input('EK-slider', 'value')
    ]
)
def update_figure(A, B, E1, E2, T1, T2, EK):
    A, B = max(A, B), min(A, B)  # Ensure A >= B
    CA = E1 * T1**A
    CB = E2 * T2**B
    config = {
        'A': A,
        'B': B,
        'CA': CA,
        'CB': CB,
        'EK': EK
    }
    # RATE = 500  # in % / s
    config_test = {
        'I': 0.15,
        'D': 0.0225,  # D = I / (RATE / 100)
        'L': 10,
    }

    fig = go.Figure()

    data = np.load('dwell_v50_key21.npz')
    t = data['t']
    f = data['f']
    fig.add_trace(go.Scattergl(
        x=t-t[0] + config_test['D'], y=f*1e9, mode='markers', marker={'color': 'blue', 'size': 4}, name='data'))

    sls = FractionalSLS(config)
    sls.set_test(config_test)
    sls.make()
    sls.run()
    dwell = sls.get_dwell()

    fig.add_trace(go.Scattergl(
        x=dwell['time'], y=dwell['stress'], mode='lines', name='fit'))
    fig.update_layout(title='Stress vs. Time',
                      xaxis_title='Time', yaxis_title='Stress')

    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8060)

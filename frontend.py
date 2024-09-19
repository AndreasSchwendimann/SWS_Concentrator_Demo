from dash import Dash, html, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import threading
import numpy as np

app = Dash(__name__)
app.config.suppress_callback_exceptions = False

app.layout = html.Div([
    html.H1(children='SwisensPoleno Particle Concentration Calculation', style={
        'textAlign': 'center', 
        'minHeight': '100px', 
        'height': 'auto', 
        'backgroundColor': '#00536c', 
        'color': 'white',  # Ensure text is readable on the dark background
        'padding': '20px',  # Optional: Add some padding for better appearance
        'margin': '0'  # Optional: Remove the margin to avoid a white border around the header
    }),
    html.Div(children=[
        html.P('Select Concentration Window:', style={'fontWeight': 'bold'}),
        dcc.RadioItems(
            options=[{'label': '1min', 'value': 1},
                {'label': '5min', 'value': 5},
                {'label': '15min', 'value': 15},
                {'label': '1h', 'value': 60},
                {'label': '2h', 'value': 120}],
            value=5, 
            id='timeframe-selector'),
        html.P('Select Binsize:', style={'fontWeight': 'bold'}),
        dcc.RadioItems(
            options=[{'label': '1s', 'value': 0.166666667},
                {'label': '1min', 'value': 1},
                {'label': '5min', 'value': 5},
                {'label': '15min', 'value': 15},
                {'label': '1h', 'value': 60},
                {'label': '2h', 'value': 120}],
            value=1, 
            id='bin-size-selector'),
        html.P('Generate Particles:', style={'fontWeight': 'bold'}),
        html.Label('Number of particles to generate:'), html.Br(),
        dcc.Input(id='nr-particles', type='number', value=5), html.Br(),
        html.Label('Timeframe to spread particles (minutes):'), html.Br(),
        dcc.Input(id='spread-timeframe', type='number', value=5), html.Br(),
        html.Label('Select time within the last 12 hours:'), html.Br(),
        dcc.Dropdown(id='time-selector', options=[], value=None), html.Br(),
        html.Button('Generate Particle Data', id='generate-particles', n_clicks=0),
        html.Button('Remove Particle Data', id='remove-particles', n_clicks=0)
    ],
    style={
        'position': 'absolute',
        'width': '20%',
        'height': 'calc(100% - 100px)',
        'top': '100px',
        'left': '0',
        'backgroundColor': '#f0f0f0',  # Optional: Add a background color for better visibility
        'padding': '10px'  # Optional: Add some padding for better appearance
    }),
    html.Div(children=[
        html.Div(id='formula-display', style={'fontSize': 56, 'textAlign': 'center', 'border': '2px solid grey', 'padding': '20px', 'margin-right': '20px'}, children=[
            html.Span(id='nr-events', style={'fontSize': 56, 'color': 'blue', 'fontWeight': 'bold'}),
            html.Span(style={'fontSize': 56, 'color': 'grey', 'fontWeight': 'bold'}, children=' / '),
            html.Span(id='volume', style={'fontSize': 56, 'color': 'green', 'fontWeight': 'bold'}),
            html.Span(style={'fontSize': 56, 'color': 'grey', 'fontWeight': 'bold'}, children=' x '),
            html.Span(style={'fontSize': 56, 'color': 'purple', 'fontWeight': 'bold'}, children='1/100%'),
            html.Span(style={'fontSize': 56, 'color': 'grey', 'fontWeight': 'bold'}, children=' x '),
            html.Span(style={'fontSize': 56, 'color': 'orange', 'fontWeight': 'bold'}, children='1/60%'),
            html.Span(style={'fontSize': 56, 'color': 'grey', 'fontWeight': 'bold'}, children=' = '),
            html.Span(id='concentration', style={'fontSize': 56, 'color': 'red', 'fontWeight': 'bold'}),
        ]), html.Br(),
        html.Div(children=[
            html.Span(style={'fontSize': 24, 'color': 'blue', 'fontWeight': 'bold'}, children="# of particles measured within timeframe"), html.Br(),
            html.Span(style={'fontSize': 24, 'color': 'green', 'fontWeight': 'bold'}, children="# of litres processed within timeframe"), html.Br(),
            html.Span(style={'fontSize': 24, 'color': 'purple', 'fontWeight': 'bold'}, children="% of particles measured and processed"), html.Br(),
            html.Span(style={'fontSize': 24, 'color': 'orange', 'fontWeight': 'bold'}, children="scaling factor to account for systematic losses"), html.Br(),
            html.Span(style={'fontSize': 24, 'color': 'red', 'fontWeight': 'bold'}, children="concentration of particles in the air within the timeframe")
        ], style={'textAlign': 'center'}),
        dcc.Graph(id='histogram'),
    ],
    style={
        'position': 'absolute',
        'width': '80%',
        'height': 'calc(100% - 100px)',
        'top': '100px',
        'left': '20%',
        'backgroundColor': '#ffffff',  # Optional: Add a background color for better visibility
        'padding': '10px'  # Optional: Add some padding for better appearance
    }),
    dcc.Interval(
        id='interval-trigger-s',
        interval=1000,  # in milliseconds
        n_intervals=0
    ),
    dcc.Store(id='dummy-output', data=0),
    dcc.Store(id='dummy-output2', data=0),
])

# Callback to remove particles
@app.callback(
    Output('dummy-output2', 'data'),
    Input('remove-particles', 'n_clicks')
)
def remove_particles(n):
    if n is None or n == 0:
        return 0
    pd.DataFrame(columns=['timestamp']).to_csv('measured_particles.csv', index=False)
    return 0

# Callback to update the time-selector dropdown
@app.callback(
    Output('dummy-output', 'data'),
    Input('generate-particles', 'n_clicks'),
    State('nr-particles', 'value'),
    State('spread-timeframe', 'value'),
    State('time-selector', 'value')
)
def generate_particles(n, nr_particles, spread_timeframe, selected_time):
    if n is None or n == 0:
        return 0
    orig = pd.read_csv('./measured_particles.csv')
    if selected_time is None:
        selected_time = pd.Timestamp.now()
    else:
        selected_time = pd.to_datetime(selected_time)
    spread_start_time = selected_time - pd.Timedelta(minutes=spread_timeframe)
    total_seconds = (selected_time - spread_start_time).total_seconds()
    
    # Randomly generate timestamps within the spread timeframe
    random_seconds = np.random.uniform(0, total_seconds, nr_particles)
    timestamps = pd.DataFrame([spread_start_time + pd.Timedelta(seconds=sec) for sec in random_seconds])
    timestamps = timestamps.apply(lambda x: x.dt.strftime('%Y-%m-%d %H:%M:%S.%f'))
    timestamps.columns = ['timestamp']
    
    exp = pd.concat([orig, timestamps], ignore_index=True)
    exp.to_csv('measured_particles.csv', index=False)
    return 0

# Callback to update the time-selector dropdown
@app.callback(
    Output('time-selector', 'options'),
    Input('interval-trigger-s', 'n_intervals')
)
def update_time_selector(n_intervals):
    now = pd.Timestamp.now()
    options = []
    for i in range(48):
        time_option = now - pd.Timedelta(minutes=15 * i)
        options.append({'label': time_option.strftime('%Y-%m-%d %H:%M'), 'value': time_option.strftime('%Y-%m-%d %H:%M')})
    return options

# Callback to update number display
@callback(
    Output('nr-events', 'children'),
    Input('interval-trigger-s', 'n_intervals'),
    Input('timeframe-selector', 'value')
)
def update_nr_events(n, timeframe):
    df = pd.read_csv('./measured_particles.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Filter the DataFrame based on the selected timeframe
    end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(minutes=timeframe)
    df_filtered = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
    
    nr_events = len(df_filtered)

    return f'{nr_events} particles'

@callback(
    Output('concentration', 'children'),
    Input('interval-trigger-s', 'n_intervals'),
    Input('timeframe-selector', 'value')
)
def update_concentration(n, timeframe):
    df = pd.read_csv('./measured_particles.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Filter the DataFrame based on the selected timeframe
    end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(minutes=timeframe)
    df_filtered = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
    
    nr_events = len(df_filtered)
    volume = timeframe * 40  # volume in litres which were collected in the selected timeframe
    concentration = np.round(nr_events / volume * 100 * 1.67, 2)

    return f'{concentration} m-3'

@callback(
    Output('volume', 'children'),
    Input('interval-trigger-s', 'n_intervals'),
    Input('timeframe-selector', 'value')
)
def update_volume(n, timeframe):
    volume = timeframe * 40  # volume in litres which were collected in the selected timeframe
    return f'{volume}L '

# Callback to update histogram
@callback(
    Output('histogram', 'figure'),
    Input('interval-trigger-s', 'n_intervals'),
    Input('timeframe-selector', 'value'),
    Input('bin-size-selector', 'value'),
    State('histogram', 'relayoutData')
)
def update_histogram(n, timeframe, bin_size, relayout_data):
    df = pd.read_csv('./measured_particles.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter the DataFrame based on the selected timeframe
    hours_shown = 2
    end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(hours=hours_shown)
    df_filtered = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
    
    nr_bins = int(np.round(hours_shown * 60 / bin_size, 0))
    
    fig = px.histogram(df_filtered, x='timestamp', nbins=nr_bins, title=f'')
    fig.update_traces(marker_line_color='black', marker_line_width=1.5, name='# of particles')

    # Add a red rectangle to highlight the selected timeframe
    fig.add_shape(
        type="rect",
        x0=end_time - pd.Timedelta(minutes=timeframe),
        x1=end_time,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        fillcolor="red",
        opacity=0.2,
        layer="below",
        line_width=0,
    )
    
    # Set the x-axis range to the last 6 hours
    fig.update_layout(xaxis_range=[start_time, end_time])
    
    # Calculate rolling concentration
    df.set_index('timestamp', inplace=True)
    df['_count'] = 1
    
    # Add a row with a count of 0 at the current timestamp
    current_time_row = pd.DataFrame({'count': [0]}, index=[pd.Timestamp.now()])
    df = pd.concat([df, current_time_row])
    
    # Resample to fill missing timestamps
    df = df.resample('1min').sum().fillna(0)
    
    rolling_window = f'{timeframe}min'
    df['rolling_count'] = df['_count'].rolling(rolling_window).apply(
        lambda x: x.sum(), raw=False
    )
    df['rolling_concentration'] = df['_count'].rolling(rolling_window).apply(
        lambda x: x.sum() / (timeframe * 40) * 100 * 1.67, raw=False
    )
    
    df.reset_index(inplace=True)

    fig.add_trace(go.Scatter(
        x=df['index'], 
        y=df['rolling_concentration'], 
        mode='lines', 
        name='Rolling Concentration', 
        line=dict(color='red'),
        yaxis='y2',
        hovertemplate=(
            'Time: %{x}<br>'
            'Rolling Concentration: %{y:.2f} particles/m<sup>3</sup><br>'
            'Number of Particles: %{customdata}'
        ),
        customdata=df['rolling_count']  # Pass the number of particles as custom data
    ))

    # Update layout to include secondary y-axis
    fig.update_layout(
        yaxis=dict(
            title="# of particles"
        ),
        yaxis2=dict(
            title="concentration / particle per m<sup>3</sup>",
            overlaying='y',
            side='right',
            showgrid=False
        )
    )

    # Check if relayout_data contains the current x-axis range
    if relayout_data and 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
        current_xaxis_range = [relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]
        # Reapply the stored x-axis range
        fig.update_layout(xaxis_range=current_xaxis_range)

    return fig


# Run the app
if __name__ == '__main__':
    app.run(debug=False)
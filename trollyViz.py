from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.integrate import cumtrapz
from scipy.fft import fft, fftfreq
from plotly.subplots import make_subplots

dfb = pd.read_csv('./csv/baseline_no_movement.csv')
side800 = pd.read_csv('./csv/800_side.csv')
side1200 = pd.read_csv('./csv/1200_side.csv')
side1600 = pd.read_csv('./csv/1600_side.csv')
forward800 = pd.read_csv('./csv/800_forward.csv')
forward1200 = pd.read_csv('./csv/1200_forward.csv')
forward1600 = pd.read_csv('./csv/1600_forward.csv')
pushingForward = pd.read_csv('./csv/pushing_forward.csv')
pushingSide = pd.read_csv('./csv/pushing_side.csv')

app = Dash(__name__)
server = app.server
def make_graph(df, name):
    dfcopy = df.copy()
    meanXnoise = dfb['x'].median()
    meanYnoise = dfb['y'].median()
    meanZnoise = dfb['z'].median()

    dfcopy['x'] = dfcopy['x'] - meanXnoise
    dfcopy['y'] = dfcopy['y'] - meanYnoise
    dfcopy['z'] = dfcopy['z'] - meanZnoise

    acc = dfcopy[['x', 'y', 'z']].values  # Acceleration data
    time = dfcopy['time'].values  # Time data
    time = time - time[0]

    velocity = cumtrapz(acc, time, initial=0, axis=0)
    position = cumtrapz(velocity, time, initial=0, axis=0)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=position[:, 0], 
        y=position[:, 1], 
        z=position[:, 2], 
        mode='lines'
    )])

    fig.update_layout(
        title=f"Trolley Trajectory with dataset: {name}",
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Z Position'
        )
    )

    return fig

# Function to calculate range with padding
def calculate_range(data, padding_percent=10):
    data_range = max(data) - min(data)
    padding = data_range * padding_percent / 100
    return [min(data) - padding, max(data) + padding]

def fft_vibrations(df, name):
    
    dfcopy = df.copy()
    # Convert the timestamp to elapsed time in seconds
    dfcopy['elapsed_time'] = dfcopy['time'] - dfcopy['time'].iloc[0]

    # Number of samples
    N = len(dfcopy)
    
    # Calculate the time period between samples
    T = dfcopy['elapsed_time'].iloc[1] - dfcopy['elapsed_time'].iloc[0]

    # Perform FFT for each axis
    fft_x = fft(dfcopy['x'].to_numpy())
    fft_y = fft(dfcopy['y'].to_numpy())
    fft_z = fft(dfcopy['z'].to_numpy())

    # Compute the frequency bins
    xf = fftfreq(N, T)[:N//2]

    # Compute the magnitude of the FFT (2/N is a normalization factor)
    magnitude_x = 2.0/N * np.abs(fft_x[:N//2])
    magnitude_y = 2.0/N * np.abs(fft_y[:N//2])
    magnitude_z = 2.0/N * np.abs(fft_z[:N//2])

    # Create subplots
    fig = make_subplots(rows=3, cols=1, subplot_titles=('FFT X-axis', 'FFT Y-axis', 'FFT Z-axis'))

    # Add traces for each axis
    fig.add_trace(go.Scatter(x=xf, y=np.log(magnitude_x), name='X-axis'), row=1, col=1)
    fig.add_trace(go.Scatter(x=xf, y=np.log(magnitude_y), name='Y-axis'), row=2, col=1)
    fig.add_trace(go.Scatter(x=xf, y=np.log(magnitude_z), name='Z-axis'), row=3, col=1)


    # Update xaxis properties
    fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="Magnitude", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude", row=3, col=1)

    # Update titles and layout
    fig.update_layout(height=900, width=800, title_text=f"Trolley Vibration Frequency Spectrum: {name}")
    return fig

def show_vibrations(df, name):
    # Calculate elapsed time in seconds from the first timestamp
    dfcopy = df.copy()
    dfcopy['elapsed_time'] = dfcopy['time'] - dfcopy['time'].iloc[0]

    # Create subplots: one plot for each acceleration axis
    fig = make_subplots(rows=3, cols=1, subplot_titles=('Acceleration X', 'Acceleration Y', 'Acceleration Z'))

    # Add traces for each axis
    fig.add_trace(go.Scatter(x=dfcopy['elapsed_time'], y=dfcopy['x'], name='X-axis'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dfcopy['elapsed_time'], y=dfcopy['y'], name='Y-axis'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dfcopy['elapsed_time'], y=dfcopy['z'], name='Z-axis'), row=3, col=1)

    # Update xaxis properties
    fig.update_xaxes(title_text="Elapsed Time (seconds)", row=3, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=2, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=3, col=1)

    # Update titles and layout
    fig.update_layout(height=900, width=800, title_text=f"Trolley Vibrations (Acceleration) Over Time: {name}")

    return fig
    
def make_animated(df, name):
    dfcopy = df.copy()
    meanXnoise = dfb['x'].median()
    meanYnoise = dfb['y'].median()
    meanZnoise = dfb['z'].median()

    dfcopy['x'] = dfcopy['x'] - meanXnoise
    dfcopy['y'] = dfcopy['y'] - meanYnoise
    dfcopy['z'] = dfcopy['z'] - meanZnoise

    acc = dfcopy[['x', 'y', 'z']].values  # Acceleration data
    time = dfcopy['time'].values  # Time data
    time = time - time[0]

    velocity = cumtrapz(acc, time, initial=0, axis=0)
    position = cumtrapz(velocity, time, initial=0, axis=0)
    px=position[:, 0]
    py=position[:, 1]
    pz=position[:, 2]

    x_range = [min(px), max(px)]
    y_range = [min(py), max(py)]
    z_range = [min(pz), max(pz)]

    fig = go.Figure(
        data=[go.Scatter3d(x=[px[0]], y=[py[0]], z=[pz[0]], mode="lines")],
        layout=go.Layout(
            title=f"Animated 3D Trajectory: {name}",
            scene=dict(
                xaxis=dict(range=x_range, autorange=False),
                yaxis=dict(range=y_range, autorange=False),
                zaxis=dict(range=z_range, autorange=False),
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position'
            ),
            updatemenus=[{
                "type": "buttons",
                "buttons": [{
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]
                }]
            }]
        )
    )

    # Adding frames to the animation
    frames = [go.Frame(data=[go.Scatter3d(x=px[:k+1], y=py[:k+1], z=pz[:k+1], mode="lines")]) for k in range(1, len(px))]

    fig.update(frames=frames)

    return fig

app.layout = html.Div([
    html.Div([
        html.H1("Robot Trajectory Dashboard", style={'textAlign':'center'}),
        html.Div([
            dcc.Graph(id='side800', figure = make_graph(side800, "side800"), style={'display': 'inline-block', 'width': '50%'}),
            dcc.Graph(id='side1200', figure = make_graph(side1200, "side1200"), style={'display': 'inline-block', 'width': '50%'}),
            dcc.Graph(id='side1600', figure = make_graph(side1600, "side1600"), style={'display': 'inline-block', 'width': '50%'})  
        ], style={'display': 'flex', 'flex-direction': 'row'}),
        html.Div([
            dcc.Graph(id='forward800', figure = make_graph(forward800, "forward800"), style={'display': 'inline-block', 'width': '50%'}),
            dcc.Graph(id='forward1200', figure = make_graph(forward1200, "forward1200"), style={'display': 'inline-block', 'width': '50%'}),
            dcc.Graph(id='forward1600', figure = make_graph(forward1600, "forward1600"), style={'display': 'inline-block', 'width': '50%'})
        ], style={'display': 'flex', 'flex-direction': 'row'}),
        html.Div([
            dcc.Graph(id='pushingForward', figure = make_graph(pushingForward, "pushingForward"), style={'display': 'inline-block', 'width': '50%'}),
            dcc.Graph(id='pushingSide', figure = make_graph(pushingSide, "pushingSide"), style={'display': 'inline-block', 'width': '50%'})
        ], style={'display': 'flex', 'flex-direction': 'row'}),
        html.Div([
            dcc.Graph(id='forward1600vibration', figure = show_vibrations(forward1600, "forward1600"), style={'display': 'inline-block', 'width': '50%'}),
            dcc.Graph(id='side1600vibration', figure = show_vibrations(side1600, "side1600"), style={'display': 'inline-block', 'width': '50%'})
        ], style={'display': 'flex', 'flex-direction': 'row'}),
        html.Div([
            dcc.Graph(id='forward1600fft', figure = fft_vibrations(forward1600, "forward1600"), style={'display': 'inline-block', 'width': '50%'}),
            dcc.Graph(id='side1600fft', figure = fft_vibrations(side1600, "side1600"), style={'display': 'inline-block', 'width': '50%'})
        ], style={'display': 'flex', 'flex-direction': 'row'}),
        dcc.Graph(id='animated', figure = make_animated(side800, "animated forward"))
    ])
])

if __name__ == '__main__':
    app.run(debug=True)
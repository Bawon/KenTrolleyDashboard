from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
#import dash_core_components as dcc
#import dash_html_components as html
import plotly.graph_objects as go
import numpy as np
from scipy.integrate import cumtrapz

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
        dcc.Graph(id='animated', figure = make_animated(side800, "animated forward"))
    ])
])

if __name__ == '__main__':
    app.run(debug=True)
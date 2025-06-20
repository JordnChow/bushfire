import dash
from dash import dcc, html, Output, Input
import plotly.express as px
import pandas as pd
from fetch_data import fetch_hotspots_australia

# Sample data
df = fetch_hotspots_australia()

# Dot map
scatter_map = px.scatter_map(
    df, lat='lat', lon='lon', hover_name='city', size='value',
    color='value', zoom=3, height=600,
    color_continuous_scale='Viridis'
)
scatter_map.update_layout(mapbox_style='carto-positron', margin={"r":0,"t":0,"l":0,"b":0})

# Choropleth-like map
choropleth_map = px.density_map(
    df, lat='lat', lon='lon', z='value', radius=50,
    center=dict(lat=39.8283, lon=-98.5795), zoom=3, height=600,
    color_continuous_scale="YlOrRd"
)
choropleth_map.update_layout(mapbox_style='carto-positron', margin={"r":0,"t":0,"l":0,"b":0})

# Dash app
app = dash.Dash(__name__)
app.title = "Map Toggle Viewer"

app.layout = html.Div([
    html.H1("Toggle Between Choropleth and Dot Map", style={"textAlign": "center"}),

    html.Div([
        html.Button("Choropleth Map", id="choropleth-btn", n_clicks=0),
        html.Button("Dot Map", id="dot-btn", n_clicks=0)
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    dcc.Graph(id="map-output")
])

@app.callback(
    Output("map-output", "figure"),
    Input("choropleth-btn", "n_clicks"),
    Input("dot-btn", "n_clicks")
)
def toggle_map(choropleth_clicks, dot_clicks):
    return scatter_map if dot_clicks > choropleth_clicks else choropleth_map

if __name__ == '__main__':
    app.run(debug=True)
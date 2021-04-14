
import pandas as pd
import numpy as np
import plotly.express as px # I would like us to do the visualizations in plotly, if possible
import sklearn as sk # for modeling
### Dashboard packages
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Data Manipulation -----------

# Modeling --------------------

# Build Visualizations --------

#Set Up dashboard -------------
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    ''')
])

#Run Dashboard
if __name__ == '__main__':
    app.run_server(debug=True)


import pandas as pd
import numpy as np
import plotly.express as px # I would like us to do the visualizations in plotly, if possible
import sklearn as sk # for modeling
### Dashboard packages
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

#Set Up dashboard
app = dash.Dash(__name__)
server = app.server

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import date

# ---------------------------------------
# To-Do:
# - Connect to Snowflake Data
# - Build Sidebar with filters
# - Make dashboard more aesthetically pleasing
# - Incoporate additional visualizations
# - Do some modeling


# Data Manipulation -----------

## Global Variables
mytitle='Shot Analysis'
tabtitle='NHL Analysis'
myheading='Hockey Analytics'
githublink='https://github.com/dva-sports-analytics/nhl-dashboard'
sourceurl='http://www.nhl.com/stats/'
image_filename = 'assets/National_Hockey_League_shield.svg'
df = pd.read_csv('data/shots.csv')
df.rename(columns={"result.secondaryType": "shot_type", "team.triCode": "team"}, inplace=True)
team_dict = [{"label": teams, "value": teams} for teams in df['team'].unique() if not pd.isna(teams)]
seasons = {int(season): season for season in df['game_id'].astype(str).str[:4].unique() if not pd.isna(season)}
shot_type = [{"label": shot_type, "value": shot_type} for shot_type in df['shot_type'].unique() if not pd.isna(shot_type)]

#------------------------------------------------------------------------------------------------------------

## Hockey Dataframe
# I took the data manipulation done by James/Brent to get the data in quickly

df.rename(columns={"result.secondaryType": "shot_type", "team.triCode": "team"}, inplace=True)
df['scored'] = df['event_type'].apply(lambda event: 1 if event == "GOAL" else 0)
df['is_rebound_attempt'] = df['time_since_last_shot'].apply(lambda x: True if x <= 5 else False)
df['shot_type'] = df['shot_type'].apply(lambda x: 'Wrist Shot' if pd.isna(x) else x)

select_df = df[['game_id', 'team', 'scored', 'distance_to_goal', 'shot_type', 'is_rebound_attempt']]
#------------------------------------------------------------------------------------------------------------


#### Shot Summary DataFrame----------------------------------------------------------------------------------
summarized_shots = select_df.groupby(['shot_type', "is_rebound_attempt"])\
         .agg({'scored':["sum", "count"], "distance_to_goal":"mean"})\
         .reset_index()

summarized_shots["Accuracy"] = summarized_shots["scored"]["sum"] / summarized_shots["scored"]["count"]

summarized_shots["Points"] = summarized_shots["scored"]["sum"]
summarized_shots["Avg_Distance_to_Goal"] = summarized_shots["distance_to_goal"]["mean"]

summarized_shots.drop(["scored", 'distance_to_goal'], axis = 1)
#------------------------------------------------------------------------------------------------------------



# Modeling --------------------



# Build Visualizations --------

shot_type_Bar = px.bar(summarized_shots, x = 'shot_type', y = "Points",
                       labels = {"shot_type":"Shot Type"},
                       color = "is_rebound_attempt",
                       title = "Points by Shot Type", 
                       hover_data=['Accuracy', 'Avg_Distance_to_Goal'])\
                  .update_xaxes(categoryorder = "total descending")
#------------------------------------------------------------------------------------------------------------

#Set Up dashboard -------------





########### Initiate the app
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/flatly/bootstrap.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle


# Add Sidebar to Page ---------------------------------------------------------------------------------------
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "15%",
    "padding": "2% 1%",
    "background-color": "#f8f9fa",
}

DASHBOARD_STYLE = {
    "margin-left": "17%",
    "margin-right": "2%",
    "padding": "2% 1%",
}
#------------------------------------------------------------------------------------------------------------


########### Set up the layout
sidebar = html.Div(children=[
    html.Center(html.Img(src=image_filename, style = {"height":'10%', "width":"40%"})),
    html.Hr(),
    html.Center(html.H2("NHL Analytics", className = "lead")),
    #Components
    #Year
    html.Br(),
    html.P("Select Data Range:"),
    dcc.RangeSlider(
        min = min(seasons.keys()),
        max = max(seasons.keys()),
        step = None,
        marks = seasons,
        value=[min(seasons.keys()),max(seasons.keys())]
    ),
    #Team
    html.Br(),
    html.P("Select Team(s):"),
    dcc.Dropdown(
        id='team-filter',
        options= team_dict,
        value = ['ANA', 'ARI', 'BOS', 'BUF', 'CAR',
                 'CBJ', 'CGY', 'CHI', 'COL', 'DAL',
                 'DET', 'EDM', 'FLA', 'LAK', 'MIN',
                 'MTL', 'NJD', 'NSH', 'NYI', 'NYR',
                 'OTT', 'PHI', 'PIT', 'SJS', 'STL',
                 'TBL', 'TOR', 'VAN', 'VGK', 'WPG',
                 'WSH'],
                 ## ADD SELECT ALL--------------------------------------------------------------------------------+++
        multi=True
    ),
    html.Div(id='dd-output-container'),
    # Shot Type Dropdown
    html.Br(),
    html.P("Select Shot Type(s):"),
    dcc.Dropdown(
        options=shot_type,
        value = ['Backhand',
                 'Deflected',
                 'Slap Shot',
                 'Snap Shot',
                 'Tip-In',
                 'Wrap-around',
                 'Wrist Shot'],
        multi=True
    ),
    #Bottom of Sidebar
    html.Br(),
    html.A('Code on Github', href=githublink, style = {'vertical-align': 'text-bottom'}),
    html.Br(),
    html.A('Data Source', href=sourceurl, style = {'vertical-align': 'text-bottom'})
    
    ],
    style = SIDEBAR_STYLE
    )
    
    
content = html.Div(children=[
    
    dcc.Graph(
        id='shotType',
        figure=shot_type_Bar
    )
    ],
    style = DASHBOARD_STYLE
)

app.layout = html.Div([content, sidebar])


# Updating Values
@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    [dash.dependencies.Input('team-filter', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)

if __name__ == '__main__':
    app.run_server()

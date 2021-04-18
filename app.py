import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
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
# All the unique Team codes + Select All button for sidebar dropdown
team_dict = [{"label": teams, "value": teams} for teams in df['team'].unique() if not pd.isna(teams)]
team_dict = [{"label": "Select All", "value": "ALL"}] + team_dict
# All the season options
seasons = {int(season): season for season in df['game_id'].astype(str).str[:4].unique() if not pd.isna(season)}
# All the unique Shot Types + Select All button for sidebar dropdown
shot_type = [{"label": shot_type, "value": shot_type} for shot_type in df['shot_type'].unique() if not pd.isna(shot_type)]
shot_type = [{"label": "Select All", "value": "ALL"}] + shot_type
#------------------------------------------------------------------------------------------------------------

## Hockey Dataframe
# I took the data manipulation done by James/Brent to get the data in quickly

df.rename(columns={"result.secondaryType": "shot_type", "team.triCode": "team"}, inplace=True)
df['scored'] = df['event_type'].apply(lambda event: 1 if event == "GOAL" else 0)
df['is_rebound_attempt'] = df['time_since_last_shot'].apply(lambda x: True if x <= 5 else False)
df['shot_type'] = df['shot_type'].apply(lambda x: 'Wrist Shot' if pd.isna(x) else x)
df['season'] = df['game_id'].astype(str).str[:4].astype(int)

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



# Build Visualizations --------:

shot_type_Bar = px.bar(summarized_shots, x = 'shot_type', y = "Points",
                       labels = {"shot_type":"Shot Type"},
                       color = "is_rebound_attempt",
                       title = "Points by Shot Type", 
                       hover_data=['Accuracy', 'Avg_Distance_to_Goal'])\
                  .update_xaxes(categoryorder = "total descending")
#------------------------------------------------------------------------------------------------------------

#Set Up dashboard ------------------------------------------------------------------------------------------:





########### Initiate the app
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/flatly/bootstrap.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle


# Add Sidebar to Page ---------------------------------------------------------------------------------------:
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "15%",
    "padding": "2% 1%",
    "background-color": "#f8f9fa",
    "overflow": "scroll"
}

DASHBOARD_STYLE = {
    "margin-left": "17%",
    "margin-right": "2%",
    "padding": "2% 1%",
}
#------------------------------------------------------------------------------------------------------------


########### Set up the layout:
#Sidebar-----------------------------------------------------------------------------------------------------:
sidebar = html.Div(children=[
    html.Center(html.Img(src=image_filename, style = {"height":'10%', "width":"40%"})),
    html.Hr(),
    html.Center(html.H2("NHL Analytics", className = "lead")),
    #Components
    #Year
    html.Br(),
    html.P("Select Data Range:"),
    dcc.RangeSlider(
        id = 'year-filter',
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
        value = [],
                 ## ADD SELECT ALL--------------------------------------------------------------------------------+++
        multi=True
    ),
    html.Div(id='dd-output-container'),
    # Shot Type Dropdown
    html.Br(),
    html.P("Select Shot Type(s):"),
    dcc.Dropdown(
        id='shot-type-filter',
        options=shot_type,
        value = [],
        multi=True
    ),
    #Bottom of Sidebar
    html.Br(),
    html.A('Code on Github', href=githublink, style = {'vertical-align': 'text-bottom'}),
    html.Br(),
    html.A('Data Source', href=sourceurl, style = {'vertical-align': 'text-bottom'}),
    html.Br()
    ],
    style = SIDEBAR_STYLE
    )
    
#Content-----------------------------------------------------------------------------------------------------:   

content = html.Div([dbc.Spinner(children = [
#Viz Tab-------------------------------------------------------------------------------------------------:
    dcc.Tabs([
        dcc.Tab(label='Visualizations', children=[
            dcc.Graph(
                id='shotType',
                figure=shot_type_Bar
            )
    
        ]),
#Data Tab-------------------------------------------------------------------------------------------------:        
        dcc.Tab(label='Data', children=[
            html.Br(),
            dash_table.DataTable(
                id='datatable-row-ids',
                columns=[
                    {'name': i, 'id': i, 'deletable': False} for i in df.columns
                    # omit the id column
                    if i != 'id'
                ],
                data=df.to_dict('records'),
                editable=False,
                filter_action="native",
                sort_action="native",
                sort_mode='multi',
                row_selectable='multi',
                row_deletable=False,
                selected_rows=[],
                page_action='native',
                page_current= 0,
                page_size= 23,
            )
        ])
    ])
    
    
    ],
    size = "md",
    color = 'primary',
    type = 'border'
    )],
    style = DASHBOARD_STYLE
)

app.layout = html.Div([content, sidebar])


# Updating Values:
@app.callback(
    dash.dependencies.Output('datatable-row-ids', 'data'),
    [dash.dependencies.Input('year-filter', 'value'),
     dash.dependencies.Input('team-filter', 'value'),
     dash.dependencies.Input('shot-type-filter', 'value')])
def filter_data(year, team, shot_type):
    
    filtered_df = df.copy()
    
    #year filter
    min_season = min(year)
    max_season = max(year)
    
    filtered_df = filtered_df.loc[(df.season >= min_season) & (df.season <= max_season)]
    
    #team filter
    
    if 'ALL' in str(team) or set(team) == set([teams for teams in df['team'].unique() if not pd.isna(teams)]) or team == []:
        pass
    else:
        filtered_df = filtered_df[df.team.isin(team)]
    
    #Shot Type Filter    
    
    if 'ALL' in str(shot_type) or set(shot_type) == set([shots for shots in df['shot_type'].unique() if not pd.isna(shots)]) or shot_type == []:
        pass
    else:
        filtered_df = filtered_df[df.shot_type.isin(shot_type)]
        
    return filtered_df.to_dict('records')

#Updating Shot Graph:
@app.callback(
    dash.dependencies.Output('shotType', 'figure'),
    [dash.dependencies.Input('datatable-row-ids', 'data')])
def update_graph(data):   
    df = pd.DataFrame(data)
    select_df = df[['game_id', 'team', 'scored', 'distance_to_goal', 'shot_type', 'is_rebound_attempt']]

    #Recreate Dataframe
    summarized_shots = select_df.groupby(['shot_type', "is_rebound_attempt"])\
         .agg({'scored':["sum", "count"], "distance_to_goal":"mean"})\
         .reset_index()

    summarized_shots["Accuracy"] = summarized_shots["scored"]["sum"] / summarized_shots["scored"]["count"]
    
    summarized_shots["Points"] = summarized_shots["scored"]["sum"]
    summarized_shots["Avg_Distance_to_Goal"] = summarized_shots["distance_to_goal"]["mean"]
    
    summarized_shots.drop(["scored", 'distance_to_goal'], axis = 1)
    
    shot_type_Bar = px.bar(summarized_shots, x = 'shot_type', y = "Points",
                       labels = {"shot_type":"Shot Type"},
                       color = "is_rebound_attempt",
                       title = "Points by Shot Type", 
                       hover_data=['Accuracy', 'Avg_Distance_to_Goal'])\
                  .update_xaxes(categoryorder = "total descending")
                  
    return shot_type_Bar


if __name__ == '__main__':
    app.run_server()

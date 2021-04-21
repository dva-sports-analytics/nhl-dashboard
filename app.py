import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from modules.DTmodel import RFClassifier
from modules.data_processing import DataProcessing
from modules.visualizations import Visualizations

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
decisionTree_path = 'assets/ShotAnalysisDecisionTree.svg'

dp = DataProcessing(filepath='./data/shots-2017-2020.csv')

df = dp.load_data()
team_dict, shot_type, periods, seasons = dp.create_dropdowns()
#------------------------------------------------------------------------------------------------------------

## Hockey Dataframe
# I took the data manipulation done by James/Brent to get the data in quickly

#df.rename(columns={"result.secondaryType": "shot_type", "team.triCode": "team"}, inplace=True)
#df['scored'] = df['event_type'].apply(lambda event: 1 if event == "GOAL" else 0)
#df['is_rebound_attempt'] = df['time_since_last_shot'].apply(lambda x: True if x <= 5 else False)
#df['shot_type'] = df['shot_type'].apply(lambda x: 'Wrist Shot' if pd.isna(x) else x)
df['season'] = df['game_id'].astype(str).str[:4].astype(int)

#------------------------------------------------------------------------------------------------------------
# Init of Vis class
vis = Visualizations(df=df)

#### Shot Summary DataFrame----------------------------------------------------------------------------------
shot_type_Bar = vis.summarized_shots_bar()
#------------------------------------------------------------------------------------------------------------
hockey_rink = vis.hockey_rink
hockey_rink_rev = vis.hockey_rink_rev
# Modeling --------------------
# Build Visualizations --------:

#Shot Distribution
shots = vis.shot_distribution_heatmap()
## Scoring Distribution Chart

score_dist = vis.score_distribution_heatmap()
#------------------------------------------------------------------------------------------------------------
# Model Random Forest
rf = RFClassifier()
rf.load_model()
rf.predict()
dt_model = rf.plot_heatmap()
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
    html.Br(),
    html.P("Select Period(s):"),
    dcc.Dropdown(
        id='period-filter',
        options=periods,
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
            dbc.Row(dbc.Col(dcc.Graph(id='shotType', figure=shot_type_Bar) #
            )),
            dbc.Row([
                
                dbc.Col(dcc.Graph(id='score_Distribution', figure=score_dist)),
                dbc.Col(dcc.Graph(id='shotDistribution', figure=shots))
                
            ])
    
        ]),
# Predictive Models -------------------------------------------------------------------------------------------------:
        dcc.Tab(label="Predictive Models" , children=[
           # TODO: Add in the predictive models in here

            html.Img(src=decisionTree_path, style = {"height":'60%', "width":"95%"}),
            html.Br(),
            # SVG on the top with the predictive models below
            dcc.Graph(id='score_Pred1', figure=dt_model, style = {"height":'90%', "width":"95%","margin-left":"auto","margin-right":"auto"})
            # dbc.Col(dcc.Graph(id='shot_pred2', figure=shots))

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
     dash.dependencies.Input('shot-type-filter', 'value'),
     dash.dependencies.Input('period-filter', 'value'),])
def filter_data(year, team, shot_type, period):
    
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
        
    #Period Filter
    if 'ALL' in str(period) or set(period) == set([period for period in df['period'].unique() if not pd.isna(period)]) or period == []:
        pass
    elif 'OT' in str(period):
        
        period.extend([4, 5, 6, 7, 8])
        filtered_df = filtered_df[df.period.isin(period)]
        
    else:
        filtered_df = filtered_df[df.period.isin(period)]
        
    return filtered_df.to_dict('records')

#Updating Shot Graph:
@app.callback(
    dash.dependencies.Output('shotType', 'figure'),
    [dash.dependencies.Input('datatable-row-ids', 'data')])
def update_shot_type(data):   
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

#Updating Shot Graph:
@app.callback(
    dash.dependencies.Output('shotDistribution', 'figure'),
    [dash.dependencies.Input('datatable-row-ids', 'data')])
def update_shot_distribution(data):   
    df = pd.DataFrame(data)
    
    #Recreate Dataframe
    shots = go.Figure()

    shots.add_trace(go.Histogram2dContour(
            x=df["x_coordinates"],
            y=df["y_coordinates"],
            z =df["scored"],
            colorscale = 'Thermal',
            xaxis = 'x',
            yaxis = 'y',
            opacity = .8,
            showscale = True,
            name="Shot Distribution",
            hovertemplate = "x: %{x}<br>y: %{y}<br>Shots: %{z}"
            ))
    
    shots.add_trace(go.Histogram(
            y = df["y_coordinates"],
            xaxis = 'x2',
            marker = dict(
                color = 'rgba(0,0,0,1)'
            ),
            name="Y-Axis Shot Histogram"
        ))
    
    shots.add_trace(go.Histogram(
            x = df["x_coordinates"],
            yaxis = 'y2',
            marker = dict(
                color = 'rgba(0,0,0,1)'
            ),
            name="X-Axis Shot Histogram"
        ))
    
    shots.add_layout_image(
            dict(
                source='data:image/png;base64,{}'.format(hockey_rink.decode()),
                xref="x",
                yref="y",
                x=20,
                y=43,
                sizex=80,
                sizey=85,
                sizing="stretch",
                layer="below")
    )
    
    shots.update_layout(
        autosize = False,
        xaxis = dict(
            zeroline = False,
            domain = [0,0.85],
            showticklabels=False,
            fixedrange = True,
            showgrid = False
        ),
        yaxis = dict(
            zeroline = False,
            domain = [0,0.85],
            showticklabels=False,
            fixedrange = True,
            showgrid = False
        ),
        xaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            fixedrange = True,
            showgrid = False
        ),
        yaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            fixedrange = True,
            showgrid = False
        ),
        height = 600,
        width = 600,
        bargap = 0,
        hovermode = 'closest',
        showlegend = False,
        title={
            'text': "NHL Shot Distirbution",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
    )
                  
    return shots


#Updating Shot Graph:
@app.callback(
    dash.dependencies.Output('score_Distribution', 'figure'),
    [dash.dependencies.Input('datatable-row-ids', 'data')])
def update_score_distribution(data):   
    
    #Recreate Dataframe
    df = pd.DataFrame(data)
    
    ## Scoring Distribution Chart
    score_dist = go.Figure()
    
    score_dist.add_trace(go.Histogram2dContour(
            x=df["x_coordinates"],
            y=df["y_coordinates"],
            z =df["scored"],
            colorscale = 'Thermal',
            xaxis = 'x',
            yaxis = 'y',
            opacity = .8,
            showscale = True,
            histfunc = "sum",
            name="Shot Distribution",
            hovertemplate = "x: %{x}<br>y: %{y}<br>Scores: %{z}"
            ))
    
    score_dist.add_trace(go.Histogram(
            y = df.loc[df["scored"] == 1]["y_coordinates"],
            xaxis = 'x2',
            
            marker = dict(
                color = 'rgba(0,0,0,1)'
            ),
            name="Y-Axis Score Histogram"
        ))
    
    score_dist.add_trace(go.Histogram(
            x = df.loc[df["scored"] == 1]["x_coordinates"],
            yaxis = 'y2',
            marker = dict(
                color = 'rgba(0,0,0,1)'
            ),
            name="X-Axis Score Histogram"
        ))
    
    score_dist.add_layout_image(
            dict(
                source='data:image/png;base64,{}'.format(hockey_rink_rev.decode()),
                xref="x",
                yref="y",
                x=100,
                y=43,
                sizex=80,
                sizey=85,
                sizing="stretch",
                layer="below")
    )
    
    score_dist.update_layout(
        autosize = False,
        xaxis = dict(
            zeroline = False,
            domain = [0.15,1],
            showticklabels=False,
            #fixedrange = True,
            showgrid = False
        ),
        yaxis = dict(
            zeroline = False,
            domain = [0,.85],
            showticklabels=False,
            fixedrange = True,
            showgrid = False
        ),
        xaxis2 = dict(
            zeroline = False,
            domain = [0,.15],
            #fixedrange = True,
            showgrid = False
        ),
        yaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            fixedrange = True,
            showgrid = False
        ),
        height = 600,
        width = 600,
        bargap = 0,
        hovermode = 'closest',
        showlegend = False,
        title={
            'text': "NHL Scoring Distirbution",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
    )
    score_dist['layout']['xaxis2']['autorange'] = "reversed"
    score_dist['layout']['xaxis']['autorange'] = "reversed"
    
    return score_dist

if __name__ == '__main__':
    app.run_server()
    # dt = DTmodel()
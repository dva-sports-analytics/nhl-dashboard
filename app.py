import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import plotly.express as px

# Data Manipulation -----------

## Global Variables

color1='darkred'
color2='orange'
mytitle='Shot Analysis'
tabtitle='NHL Analysis'
myheading='Hockey Analytics'
label1='IBU'
label2='ABV'
githublink='https://github.com/dva-sports-analytics/nhl-dashboard'
sourceurl='http://www.nhl.com/stats/'


## Hockey Dataframe
# I took the data manipulation done by James/Brent to get the data in quickly
df = pd.read_csv('data/shots.csv')
df.rename(columns={"result.secondaryType": "shot_type", "team.triCode": "team"}, inplace=True)
df['scored'] = df['event_type'].apply(lambda event: 1 if event == "GOAL" else 0)
df['is_rebound_attempt'] = df['time_since_last_shot'].apply(lambda x: True if x <= 5 else False)
df['shot_type'] = df['shot_type'].apply(lambda x: 'Wrist Shot' if pd.isna(x) else x)

select_df = df[['game_id', 'team', 'scored', 'distance_to_goal', 'shot_type', 'is_rebound_attempt']]


beers=['Chesapeake Stout', 'Snake Dog IPA', 'Imperial Porter', 'Double Dog IPA']
ibu_values=[35, 60, 85, 75]
abv_values=[5.4, 7.1, 9.2, 4.3]


#### Shot Summary DataFrame

summarized_shots = select_df.groupby(['shot_type', "is_rebound_attempt"])\
         .agg({'scored':["sum", "count"], "distance_to_goal":"mean"})\
         .reset_index()

summarized_shots["Accuracy"] = summarized_shots["scored"]["sum"] / summarized_shots["scored"]["count"]

summarized_shots["Points"] = summarized_shots["scored"]["sum"]
summarized_shots["Avg_Distance_to_Goal"] = summarized_shots["distance_to_goal"]["mean"]

summarized_shots.drop(["scored", 'distance_to_goal'], axis = 1)

# Modeling --------------------



# Build Visualizations --------


bitterness = go.Bar(
    x=beers,
    y=ibu_values,
    name=label1,
    marker={'color':color1}
)
alcohol = go.Bar(
    x=beers,
    y=abv_values,
    name=label2,
    marker={'color':color2}
)

beer_data = [bitterness, alcohol]
beer_layout = go.Layout(
    barmode='group',
    title = mytitle
)

beer_fig = go.Figure(data=beer_data, layout=beer_layout)


shot_type_Bar = px.bar(summarized_shots, x = 'shot_type', y = "Points",
                       labels = {"shot_type":"Shot Type"},
                       color = "is_rebound_attempt",
                       title = "Points by Shot Type", 
                       hover_data=['Accuracy', 'Avg_Distance_to_Goal'])\
                  .update_xaxes(categoryorder = "total descending")


#Set Up dashboard -------------





########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

########### Set up the layout
app.layout = html.Div(children=[
    html.H1(myheading),
    dcc.Graph(
        id='shotType',
        figure=shot_type_Bar
    ),
    html.A('Code on Github', href=githublink),
    html.Br(),
    html.A('Data Source', href=sourceurl),
    ]
)

if __name__ == '__main__':
    app.run_server()

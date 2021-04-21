

# In[8]:


from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go
import  base64
hockey_rink_rev_filepath = 'assets/Half_ice_hockey_rink_rev.png'
hockey_rink_rev = base64.b64encode(open(hockey_rink_rev_filepath, 'rb').read())
def DTmodel():

    df = pd.read_csv('./data/shots-2017-2020.csv')
    #df.rename(columns={"result.secondaryType": "shot_type", "team.triCode": "team"}, inplace=True)

    #df['scored'] = df['event_type'].apply(lambda event: 1 if event == "GOAL" else 0)
    #df['is_rebound_attempt'] = df['time_since_last_shot'].apply(lambda x: True if x <= 5 else False)
    #df['shot_type'] = df['shot_type'].apply(lambda x: 'Wrist Shot' if pd.isna(x) else x)

    df.loc[df.period <= 3, "total_time_remaining"] = (3 - df.loc[df.period <= 3]['period']) * 1200 + df.loc[df.period <= 3]['period_time_remaining']
    df.loc[df.period > 3, "total_time_remaining"] = 0

    scaler = MinMaxScaler()
    scaler.fit(df[['period_time_remaining', 'distance_to_goal', 'x_coordinates', "y_coordinates", 'total_time_remaining', 'time_of_last_shot', 'time_since_last_shot']])
    df[['period_time_remaining', 'distance_to_goal', 'x_coordinates', "y_coordinates", 'total_time_remaining', 'time_of_last_shot', 'time_since_last_shot']] = scaler.transform(
        df[['period_time_remaining', 'distance_to_goal', 'x_coordinates', "y_coordinates", 'total_time_remaining', 'time_of_last_shot', 'time_since_last_shot']])

    cat_vars=['team', 'shot_type','is_rebound_attempt']
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(df[var], prefix=var)
        df1=df.join(cat_list)
        df=df1
    cat_vars=['team', 'shot_type','is_rebound_attempt']
    data_vars=df.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_vars]

    df = df[to_keep]
    df = df[['period', 'period_time_remaining', 'coordinates.x',
           'coordinates.y', 'distance_to_goal',
           'time_of_last_shot', 'time_since_last_shot',
           'scored', 'total_time_remaining', 'team_ANA', 'team_ARI',
           'team_BOS', 'team_BUF', 'team_CAR', 'team_CBJ', 'team_CGY',
           'team_CHI', 'team_COL', 'team_DAL', 'team_DET', 'team_EDM',
           'team_FLA', 'team_LAK', 'team_MIN', 'team_MTL', 'team_NJD',
           'team_NSH', 'team_NYI', 'team_NYR', 'team_OTT', 'team_PHI',
           'team_PIT', 'team_SJS', 'team_STL', 'team_TBL', 'team_TOR',
           'team_VAN', 'team_VGK', 'team_WPG', 'team_WSH',
           'shot_type_Backhand', 'shot_type_Deflected', 'shot_type_Slap Shot',
           'shot_type_Snap Shot', 'shot_type_Tip-In', 'shot_type_Wrap-around',
           'shot_type_Wrist Shot', 'is_rebound_attempt_False',
           'is_rebound_attempt_True']]

    df = df.dropna(subset=['time_since_last_shot', 'time_of_last_shot'])
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'scored'], df[['scored']], test_size=0.33, random_state=42)
    columns = X_train.columns

    oversample = SMOTE(random_state=42)

    X_train, y_train = oversample.fit_resample(X_train, y_train)

    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train.values.ravel())
    print("Predicting...")
    pred = clf.predict(X_test)
    print("scoring")
    print(clf.score(X_test, y_test))

    print(confusion_matrix(y_test, pred))
    con_mat = confusion_matrix(y_test, pred)
    print("Predictive Value:")
    specificity = con_mat[1][1] / (con_mat[1][1] + con_mat[0][1])

    Z_probs = clf.predict_proba(X_test)
    X_test_final = X_test.copy()
    X_test_final['scoreProb'] = [i[1] for i in Z_probs]


    # In[9]:


    score_probs = go.Figure()

    score_probs.add_trace(go.Histogram2dContour(
            x=X_test_final["x_coordinates"],
            y=X_test_final["y_coordinates"],
            z =X_test_final["scoreProb"]*specificity,
            colorscale = 'Thermal',
            xaxis = 'x',
            yaxis = 'y',
            opacity = .8,
            showscale = True,
            histfunc = "avg",
            name="Shot Distribution",
            hovertemplate = "x: %{x}<br>y: %{y}<br>Probability: %{z}"
            ))

    score_probs.add_trace(go.Bar(
            y = X_test_final.groupby('y_coordinates').agg({'scoreProb': 'mean'}).reset_index()['y_coordinates'],
            x = X_test_final.groupby('x_coordinates').agg({'scoreProb': 'mean'}).reset_index()["scoreProb"]*.15,
            xaxis = 'x2',
            orientation='h',
            marker = dict(
                color = 'rgba(0,0,0,1)'
            ),
            hovertemplate = "y: %{y}<br>Probability: %{x}",
            name="Y-Axis Score Histogram"
        ))

    score_probs.add_trace(go.Histogram(
            x = X_test_final["x_coordinates"],
            y = X_test_final["scoreProb"]*specificity,
            histfunc = 'avg',
            yaxis = 'y2',
            nbinsx=100,
            marker = dict(
                color = 'rgba(0,0,0,1)'
            ),
            hovertemplate = "y: %{x}<br>Probability: %{y}",
            name="X-Axis Score Histogram"
        ))

    score_probs.update_layout(
        autosize = False,
        xaxis = dict(
            zeroline = False,
            domain = [0.15,1],
            showticklabels=False,
            fixedrange = True,
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
            'text': "NHL Predicted Scoring Probabilities",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
    )
    score_probs['layout']['xaxis2']['autorange'] = "reversed"
    score_probs['layout']['xaxis']['autorange'] = "reversed"
    score_probs.add_layout_image(
               dict(
                   source='data:image/png;base64,{}'.format(hockey_rink_rev.decode()),
                   xref="x",
                   yref="y",
                   x=1.07,
                   y=1.06,
                   sizex=1.1,
                   sizey=1.1,
                   sizing="stretch",
                   layer="below")
       )
    return score_probs







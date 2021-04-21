import base64

import plotly.express as px
import plotly.graph_objs as go


class Visualizations():

	def __init__(self, df):
		self.df = df
		self.hockey_rink_filepath = 'assets/Half_ice_hockey_rink.png'
		self.hockey_rink = base64.b64encode(open(self.hockey_rink_filepath, 'rb').read())
		self.hockey_rink_rev_filepath = 'assets/Half_ice_hockey_rink_rev.png'
		self.hockey_rink_rev = base64.b64encode(open(self.hockey_rink_rev_filepath, 'rb').read())

	def summarized_shots_bar(self):
		select_df = self.df[['game_id', 'team', 'scored', 'distance_to_goal', 'shot_type', 'is_rebound_attempt']]
		summarized_shots = select_df.groupby(['shot_type', "is_rebound_attempt"]) \
			.agg({'scored': ["sum", "count"], "distance_to_goal": "mean"}) \
			.reset_index()

		summarized_shots["Accuracy"] = summarized_shots["scored"]["sum"] / summarized_shots["scored"]["count"]

		summarized_shots["Points"] = summarized_shots["scored"]["sum"]
		summarized_shots["Avg_Distance_to_Goal"] = summarized_shots["distance_to_goal"]["mean"]

		summarized_shots.drop(["scored", 'distance_to_goal'], axis=1)
		self.shot_type_Bar = px.bar(summarized_shots, x='shot_type', y="Points",
		                       labels={"shot_type": "Shot Type"},
		                       color="is_rebound_attempt",
		                       title="Points by Shot Type",
		                       hover_data=['Accuracy', 'Avg_Distance_to_Goal']) \
			.update_xaxes(categoryorder="total descending")
		return self.shot_type_Bar

	def shot_distribution_heatmap(self):
		# hockey_rink = 'assets/Half_ice_hockey_rink.png'
		hockey_rink = self.hockey_rink

		shots = go.Figure()

		shots.add_trace(go.Histogram2dContour(
			x=self.df["x_coordinates"],
			y=self.df["y_coordinates"],
			z=self.df["scored"],
			colorscale='Thermal',
			xaxis='x',
			yaxis='y',
			opacity=.8,
			showscale=True,
			name="Shot Distribution",
			hovertemplate="x: %{x}<br>y: %{y}<br>Shots: %{z}"
		))

		shots.add_trace(go.Histogram(
			y=self.df["y_coordinates"],
			xaxis='x2',
			marker=dict(
				color='rgba(0,0,0,1)'
			),
			name="Y-Axis Shot Histogram"
		))

		shots.add_trace(go.Histogram(
			x=self.df["x_coordinates"],
			yaxis='y2',
			marker=dict(
				color='rgba(0,0,0,1)'
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
			autosize=False,
			xaxis=dict(
				zeroline=False,
				domain=[0, 0.85],
				showticklabels=False,
				fixedrange=True,
				showgrid=False
			),
			yaxis=dict(
				zeroline=False,
				domain=[0, 0.85],
				showticklabels=False,
				fixedrange=True,
				showgrid=False
			),
			xaxis2=dict(
				zeroline=False,
				domain=[0.85, 1],
				fixedrange=True,
				showgrid=False
			),
			yaxis2=dict(
				zeroline=False,
				domain=[0.85, 1],
				fixedrange=True,
				showgrid=False
			),
			height=600,
			width=600,
			bargap=0,
			hovermode='closest',
			showlegend=False,
			title={
				'text': "NHL Shot Distirbution",
				'y': 0.9,
				'x': 0.5,
				'xanchor': 'center',
				'yanchor': 'top'}
		)
		self.shots = shots
		return self.shots

	def score_distribution_heatmap(self):


		score_dist = go.Figure()

		score_dist.add_trace(go.Histogram2dContour(
			x=self.df["x_coordinates"],
			y=self.df["y_coordinates"],
			z=self.df["scored"],
			colorscale='Thermal',
			xaxis='x',
			yaxis='y',
			opacity=.8,
			showscale=True,
			histfunc="sum",
			name="Shot Distribution",
			hovertemplate="x: %{x}<br>y: %{y}<br>Scores: %{z}"
		))

		score_dist.add_trace(go.Histogram(
			y=self.df.loc[self.df["scored"] == 1]["y_coordinates"],
			xaxis='x2',

			marker=dict(
				color='rgba(0,0,0,1)'
			),
			name="Y-Axis Score Histogram"
		))

		score_dist.add_trace(go.Histogram(
			x=self.df.loc[self.df["scored"] == 1]["x_coordinates"],
			yaxis='y2',
			marker=dict(
				color='rgba(0,0,0,1)'
			),
			name="X-Axis Score Histogram"
		))

		score_dist.add_layout_image(
			dict(
				source='data:image/png;base64,{}'.format(self.hockey_rink_rev.decode()),
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
			autosize=False,
			xaxis=dict(
				zeroline=False,
				domain=[0.15, 1],
				showticklabels=False,
				# fixedrange = True,
				showgrid=False
			),
			yaxis=dict(
				zeroline=False,
				domain=[0, .85],
				showticklabels=False,
				fixedrange=True,
				showgrid=False
			),
			xaxis2=dict(
				zeroline=False,
				domain=[0, .15],
				# fixedrange = True,
				showgrid=False
			),
			yaxis2=dict(
				zeroline=False,
				domain=[0.85, 1],
				fixedrange=True,
				showgrid=False
			),
			height=600,
			width=600,
			bargap=0,
			hovermode='closest',
			showlegend=False,
			title={
				'text': "NHL Scoring Distirbution",
				'y': 0.9,
				'x': 0.5,
				'xanchor': 'center',
				'yanchor': 'top'}
		)
		score_dist['layout']['xaxis2']['autorange'] = "reversed"
		score_dist['layout']['xaxis']['autorange'] = "reversed"
		self.score_dist = score_dist
		return self.score_dist

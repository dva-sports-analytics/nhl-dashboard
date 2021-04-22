import pandas as pd
import os



class DataProcessing():

	def __init__(self, filepath = 'data/shots-2017-2020_small.csv'):
		print('init data processing')
		self.filepath = filepath

	def load_data(self):
		print(f'loading file {self.filepath}')
		self.df = pd.read_csv(self.filepath)
		self.df.loc[self.df.period <= 3, "total_time_remaining"] = (3 - self.df.loc[self.df.period <= 3]['period']) * 1200 +  self.df.loc[self.df.period <= 3]['period_time_remaining']
		self.df.loc[self.df.period > 3, "total_time_remaining"] = 0
		#self.df.rename(columns={"result.secondaryType": "shot_type", "team.triCode": "team"}, inplace=True)
		return self.df
	def create_dropdowns(self):
	    # All the unique Team codes + Select All button for sidebar dropdown
	    print('Creating Dropdown Labels')
	    team_dict = [{"label": teams, "value": teams} for teams in self.df['team'].unique() if not pd.isna(teams)]
	    team_dict = [{"label": "Select All", "value": "ALL"}] + team_dict
	    # All the season options
	    seasons = {int(season): season for season in self.df['game_id'].astype(str).str[:4].unique() if not pd.isna(season)}
	    # All the unique Shot Types + Select All button for sidebar dropdown
	    shot_type = [{"label": shot_type, "value": shot_type} for shot_type in self.df['shot_type'].unique() if not pd.isna(shot_type)]
	    shot_type = [{"label": "Select All", "value": "ALL"}] + shot_type
	    # All the Period Options
	    periods = [{"label": str(period), "value": str(period)} for period in self.df['period'].unique() if not pd.isna(period) and int(period) <= 3]
	    periods = [{"label": "Select All", "value": "ALL"}] + periods + [{"label": "Overtime", "value": "OT"}]
	    return team_dict, shot_type, periods, seasons



if __name__ == '__main__':
	print('Testing Data Processing')

	dp = DataProcessing(filepath='data/shots-2017-2020_small.csv')
	df = dp.load_data()
	print(df.head())
	team_dict, shot_type, periods, seasons = dp.create_dropdowns()
	print(team_dict)
	print(shot_type)
	print(periods)
	print(seasons)

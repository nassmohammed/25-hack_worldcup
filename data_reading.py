import pandas as pd
import numpy as np

matches_df = pd.read_csv('matches_2010_2022.csv')
elos_df = pd.read_csv('team_elo_sums.csv')

X = []
y = []
for home, away, date, result in zip(matches_df['home_team_name'], matches_df['away_team_name'], matches_df['match_date'], matches_df['result']):
    match_year = int(date.split('-')[0])
    wc_df = elos_df[elos_df['year'] == match_year]
    home_features = wc_df[['elo_sum_GK', 'elo_sum_DF', 'elo_sum_MF', 'elo_sum_FW']][wc_df['team'] == home]
    away_features = wc_df[['elo_sum_GK', 'elo_sum_DF', 'elo_sum_MF', 'elo_sum_FW']][wc_df['team'] == away]
    
    X.append(np.concatenate([home_features.values[0], away_features.values[0]]))
    
    y.append([1, 0, 0] if result == 'home team win' else [0, 1, 0] if result == 'draw' else [0, 0, 1])

    
    
    
    

    
    
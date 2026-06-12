import pandas as pd
import numpy as np

matches_df = pd.read_csv('matches_2010_2022.csv')
elos_df = pd.read_csv('team_elo_sums_long.csv')

import numpy as np

POS = ['GK', 'DF', 'MF', 'FW']

def avg_features(row):
    feats = []
    for p in POS:
        n = row[f'n_{p}_with_elo']
        feats.append(row[f'elo_sum_{p}'] / n if n > 0 else 0.0)
    return np.array(feats, dtype=float)

X = []
y = []
for home, away, date, result in zip(matches_df['home_team_name'], matches_df['away_team_name'], matches_df['match_date'], matches_df['result']):
    match_year = int(date.split('-')[0])
    wc_df = elos_df[elos_df['year'] == match_year]

    home_row = wc_df[wc_df['team'] == home]
    away_row = wc_df[wc_df['team'] == away]

    home_features = avg_features(home_row.iloc[0])
    away_features = avg_features(away_row.iloc[0])

    X.append(np.concatenate([home_features, away_features]))

    y.append([1, 0, 0] if result == 'home team win' else [0, 1, 0] if result == 'draw' else [0, 0, 1])

features = np.array(X)
labels = np.array(y)

np.save('features_norm.npy', features)
np.save('labels_norm.npy', labels)



    
    
    
    

    
    
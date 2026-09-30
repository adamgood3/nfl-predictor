import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import nfl_data_py as nfl
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

print("1. Ingesting live 2026 data and master schedule...")
pbp = nfl.import_pbp_data([2026])
players = nfl.import_players()
schedule = nfl.import_schedules([2026])

# Filter down to offensive snaps from skill positions
skill_players = players[players['position'].isin(['WR', 'TE', 'RB'])][['gsis_id', 'position']]
passes = pbp[pbp['play_type'] == 'pass'].merge(
    skill_players, left_on='receiver_player_id', right_on='gsis_id', how='inner'
)

# ---------------------------------------------------------
# HISTORICAL WEEKLY TOTALS
# ---------------------------------------------------------
team_wk_totals = passes.groupby(['posteam', 'week'])['play_id'].count().reset_index(name='team_passes')
pos_wk_totals = passes.groupby(['posteam', 'defteam', 'week', 'position'])['play_id'].count().reset_index(name='pos_targets')

weekly_shares = pos_wk_totals.merge(team_wk_totals, on=['posteam', 'week'])
weekly_shares['actual_tgt_sh'] = weekly_shares['pos_targets'] / weekly_shares['team_passes']

# ---------------------------------------------------------
# CONSTRUCT WEEKS 1-3 BASELINES FOR TRAINING & INFERENCE
# ---------------------------------------------------------
past_passes = passes[passes['week'] <= 3].copy()

off_tot = past_passes.groupby('posteam')['play_id'].count().reset_index(name='off_tot_passes')
off_pos = past_passes.groupby(['posteam', 'position'])['play_id'].count().reset_index(name='off_pos_targets')
off_profile = off_pos.merge(off_tot, on='posteam')
off_profile['off_tgt_sh'] = off_profile['off_pos_targets'] / off_profile['off_tot_passes']

def_tot = past_passes.groupby('defteam')['play_id'].count().reset_index(name='def_tot_passes')
def_pos = past_passes.groupby(['defteam', 'position'])['play_id'].count().reset_index(name='def_pos_targets')
def_profile = def_pos.merge(def_tot, on='defteam')
def_profile['def_tgt_sh_allowed'] = def_profile['def_pos_targets'] / def_profile['def_tot_passes']

# ---------------------------------------------------------
# COMPILE TRAINING SET & TRAIN MODEL
# ---------------------------------------------------------
print("2. Building feature set and training Gradient Boosting model...")
train_df = weekly_shares[weekly_shares['week'].isin([2, 3])].merge(
    off_profile[['posteam', 'position', 'off_tgt_sh']], on=['posteam', 'position'], how='inner'
).merge(
    def_profile[['defteam', 'position', 'def_tgt_sh_allowed']], on=['defteam', 'position'], how='inner'
)

X_train = train_df[['off_tgt_sh', 'def_tgt_sh_allowed']]
y_train = train_df['actual_tgt_sh']

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# GENERATE CLEAN PREDICTIONS FOR WEEK 4 MATCHUPS
# ---------------------------------------------------------
print("3. Generating predictions for Week 4 slate...")
wk4_games = schedule[schedule['week'] == 4][['game_id', 'home_team', 'away_team']].dropna()

# Evaluate Away Offenses (vs Home Defenses)
away_eval = wk4_games.merge(off_profile, left_on='away_team', right_on='posteam').merge(
    def_profile, left_on=['home_team', 'position'], right_on=['defteam', 'position']
)
away_eval['Offense'] = away_eval['away_team']
away_eval['Defense'] = away_eval['home_team']

# Evaluate Home Offenses (vs Away Defenses)
home_eval = wk4_games.merge(off_profile, left_on='home_team', right_on='posteam').merge(
    def_profile, left_on=['away_team', 'position'], right_on=['defteam', 'position']
)
home_eval['Offense'] = home_eval['home_team']
home_eval['Defense'] = home_eval['away_team']

# Combine both sides of the ball into one full slate
full_eval = pd.concat([away_eval, home_eval], ignore_index=True)

full_eval['predicted_tgt_sh'] = model.predict(full_eval[['off_tgt_sh', 'def_tgt_sh_allowed']])
full_eval['script_delta'] = full_eval['predicted_tgt_sh'] - full_eval['off_tgt_sh']
full_eval['Game'] = full_eval['away_team'] + " @ " + full_eval['home_team']

# Select columns
cols = ['Game', 'Offense', 'Defense', 'position', 'off_tgt_sh', 'def_tgt_sh_allowed', 'predicted_tgt_sh', 'script_delta']
display_df = full_eval[cols].copy()

# ---> THIS IS THE NEW LINE: Sort by Delta from highest to lowest BEFORE converting to strings
display_df = display_df.sort_values(by='script_delta', ascending=False)

# Convert long decimals to clean percentages (e.g. 0.175 -> 17.5%)
for col in ['off_tgt_sh', 'def_tgt_sh_allowed', 'predicted_tgt_sh', 'script_delta']:
    display_df[col] = (display_df[col] * 100).round(1).astype(str) + '%'

display_df = display_df.rename(columns={
    'position': 'Pos',
    'off_tgt_sh': 'Off_Avg',
    'def_tgt_sh_allowed': 'Def_Allows',
    'predicted_tgt_sh': 'Projected',
    'script_delta': 'Delta'
})

print("\n=== WEEK 4 MATCHUP PROJECTIONS ===")
print(display_df.head(10).to_string(index=False))

# Save the final clean version
display_df.to_csv("weekly_predictions.csv", index=False)
print("\nSuccess: Saved clean predictions to weekly_predictions.csv")
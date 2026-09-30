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
# GENERATE PREDICTIONS FOR WEEK 4
# ---------------------------------------------------------
print("3. Generating predictions for Week 4 slate...")
wk4_games = schedule[schedule['week'] == 4][['game_id', 'home_team', 'away_team']].dropna()

away_eval = wk4_games.merge(off_profile, left_on='away_team', right_on='posteam').merge(
    def_profile, left_on=['home_team', 'position'], right_on=['defteam', 'position']
)
away_eval['predicted_tgt_sh'] = model.predict(away_eval[['off_tgt_sh', 'def_tgt_sh_allowed']])
away_eval['script_delta'] = away_eval['predicted_tgt_sh'] - away_eval['off_tgt_sh']
away_eval['matchup'] = away_eval['away_team'] + " @ " + away_eval['home_team']

cols = ['matchup', 'position', 'off_tgt_sh', 'def_tgt_sh_allowed', 'predicted_tgt_sh', 'script_delta']
display_df = away_eval[cols].copy()
display_df = display_df.rename(columns={
    'off_tgt_sh': 'Off_Baseline',
    'def_tgt_sh_allowed': 'Def_Allowed',
    'predicted_tgt_sh': 'Projected_Share',
    'script_delta': 'Matchup_Delta'
})

print("\n=== WEEK 4 GAME SCRIPT PREDICTIONS (AWAY TEAMS) ===")
print(display_df.sort_values(by='Matchup_Delta', ascending=False).head(12).to_string(index=False))

# --- NEW STEP FOR GITHUB: SAVE AS CSV ---
display_df.sort_values(by='Matchup_Delta', ascending=False).to_csv("weekly_predictions.csv", index=False)
print("\nSuccess: Saved predictions to weekly_predictions.csv")
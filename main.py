import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import nfl_data_py as nfl
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

print("1. Ingesting live 2026 data, schedules, and depth charts...")
pbp = nfl.import_pbp_data([2026])
schedule = nfl.import_schedules([2026])
depth_charts = nfl.import_depth_charts([2026])

# ---------------------------------------------------------
# INTEGRATE OFFICIAL DEPTH CHARTS FOR EXACT ROLES
# ---------------------------------------------------------
print("2. Mapping structural roles (WR1, WR2, TE1, etc.)...")

# NEW SCHEMA FIX: 2025+ uses 'pos_abb' instead of 'position', 'pos_rank' instead of 'depth_team'
dc_filtered = depth_charts[depth_charts['pos_abb'].isin(['WR', 'TE', 'RB'])].copy()

# Ensure pos_rank is numeric
dc_filtered['pos_rank'] = pd.to_numeric(dc_filtered['pos_rank'], errors='coerce')
dc_filtered = dc_filtered.dropna(subset=['pos_rank'])

# Sort by the 'dt' timestamp to get the most up-to-date depth chart for players
if 'dt' in dc_filtered.columns:
    dc_filtered = dc_filtered.sort_values(by='dt', ascending=False)

def assign_role(row):
    pos = row['pos_abb']
    depth = int(row['pos_rank'])
    
    if pos == 'WR' and depth <= 3:
        return f"WR{depth}"
    elif pos in ['TE', 'RB'] and depth <= 2:
        return f"{pos}{depth}"
    return None

dc_filtered['role'] = dc_filtered.apply(assign_role, axis=1)
dc_filtered = dc_filtered.dropna(subset=['role'])

# Keep only unique players and their most recent role
dc_roles = dc_filtered[['gsis_id', 'role']].drop_duplicates(subset=['gsis_id'])

# Merge roles into play-by-play
passes = pbp[pbp['play_type'] == 'pass'].copy()
passes = passes.merge(dc_roles, left_on='receiver_player_id', right_on='gsis_id', how='inner')

# ---------------------------------------------------------
# HISTORICAL WEEKLY TOTALS BY ROLE
# ---------------------------------------------------------
print("3. Calculating weekly defensive vulnerabilities by role...")
team_wk_totals = passes.groupby(['posteam', 'week'])['play_id'].count().reset_index(name='team_passes')
role_wk_totals = passes.groupby(['posteam', 'defteam', 'week', 'role'])['play_id'].count().reset_index(name='role_targets')

weekly_shares = role_wk_totals.merge(team_wk_totals, on=['posteam', 'week'])
weekly_shares['actual_tgt_sh'] = weekly_shares['role_targets'] / weekly_shares['team_passes']

# ---------------------------------------------------------
# CONSTRUCT WEEKS 1-3 BASELINES FOR TRAINING & INFERENCE
# ---------------------------------------------------------
past_passes = passes[passes['week'] <= 3].copy()

off_tot = past_passes.groupby('posteam')['play_id'].count().reset_index(name='off_tot_passes')
off_role = past_passes.groupby(['posteam', 'role'])['play_id'].count().reset_index(name='off_role_targets')
off_profile = off_role.merge(off_tot, on='posteam')
off_profile['off_tgt_sh'] = off_profile['off_role_targets'] / off_profile['off_tot_passes']

def_tot = past_passes.groupby('defteam')['play_id'].count().reset_index(name='def_tot_passes')
def_role = past_passes.groupby(['defteam', 'role'])['play_id'].count().reset_index(name='def_role_targets')
def_profile = def_role.merge(def_tot, on='defteam')
def_profile['def_tgt_sh_allowed'] = def_profile['def_role_targets'] / def_profile['def_tot_passes']

# ---------------------------------------------------------
# COMPILE TRAINING SET & TRAIN MODEL
# ---------------------------------------------------------
print("4. Building feature set and training Gradient Boosting model...")
train_df = weekly_shares[weekly_shares['week'].isin([2, 3])].merge(
    off_profile[['posteam', 'role', 'off_tgt_sh']], on=['posteam', 'role'], how='inner'
).merge(
    def_profile[['defteam', 'role', 'def_tgt_sh_allowed']], on=['defteam', 'role'], how='inner'
)

X_train = train_df[['off_tgt_sh', 'def_tgt_sh_allowed']]
y_train = train_df['actual_tgt_sh']

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# GENERATE CLEAN PREDICTIONS FOR WEEK 4 MATCHUPS
# ---------------------------------------------------------
print("5. Generating role-specific predictions for Week 4 slate...")
wk4_games = schedule[schedule['week'] == 4][['game_id', 'home_team', 'away_team']].dropna()

away_eval = wk4_games.merge(off_profile, left_on='away_team', right_on='posteam').merge(
    def_profile, left_on=['home_team', 'role'], right_on=['defteam', 'role']
)
away_eval['Offense'] = away_eval['away_team']
away_eval['Defense'] = away_eval['home_team']

home_eval = wk4_games.merge(off_profile, left_on='home_team', right_on='posteam').merge(
    def_profile, left_on=['away_team', 'role'], right_on=['defteam', 'role']
)
home_eval['Offense'] = home_eval['home_team']
home_eval['Defense'] = home_eval['away_team']

full_eval = pd.concat([away_eval, home_eval], ignore_index=True)

full_eval['predicted_tgt_sh'] = model.predict(full_eval[['off_tgt_sh', 'def_tgt_sh_allowed']])
full_eval['script_delta'] = full_eval['predicted_tgt_sh'] - full_eval['off_tgt_sh']
full_eval['Game'] = full_eval['away_team'] + " @ " + full_eval['home_team']

cols = ['Game', 'Offense', 'Defense', 'role', 'off_tgt_sh', 'def_tgt_sh_allowed', 'predicted_tgt_sh', 'script_delta']
display_df = full_eval[cols].copy().sort_values(by='script_delta', ascending=False)

for col in ['off_tgt_sh', 'def_tgt_sh_allowed', 'predicted_tgt_sh', 'script_delta']:
    display_df[col] = (display_df[col] * 100).round(1).astype(str) + '%'

display_df = display_df.rename(columns={
    'role': 'Pos',
    'off_tgt_sh': 'Off_Avg',
    'def_tgt_sh_allowed': 'Def_Allows',
    'predicted_tgt_sh': 'Projected',
    'script_delta': 'Delta'
})

print("\n=== WEEK 4 ROLE-SPECIFIC MATCHUP PROJECTIONS ===")
print(display_df.head(10).to_string(index=False))

display_df.to_csv("weekly_predictions.csv", index=False)
print("\nSuccess: Saved role-specific predictions to weekly_predictions.csv")
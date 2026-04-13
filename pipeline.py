# ============================================================
# pipeline.py
#
# Reusable functions for the serve-receive set prediction project. 
#
# Functions:
#   load_and_combine_games()    — loop dvw folder, run R script, combine CSVs
#   build_features()            — encoding, rotation mapping, lag features
#   mask_probabilities()        — zero out players not on court, renormalize
#   compute_predictability()    — entropy, top choice rate, rotation breakdown
# ============================================================

import os
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy
from sklearn.preprocessing import LabelEncoder

# 1. DATA LOADING

def load_and_combine_games(
    dvw_folder,
    output_folder,
    r_script_path,
    focus_team_id,
    focus_team_name,
    force_rerun=False
):
    """
    Loop through all .dvw files in dvw_folder, run extract_serve_receive.R
    on each, and combine into a single DataFrame.

    Filename convention expected:
        {game_num}_{period}_{split}_vs_{opponent}.dvw
        e.g. 01_pre_train_vs_Stanford.dvw

    Parameters:
        dvw_folder      : path to folder containing .dvw files
        output_folder   : path to save per-game CSVs
        r_script_path   : path to extract_serve_receive.R
        focus_team_id   : numeric team ID (string ok)
        focus_team_name : team name string as it appears in dvw data
        force_rerun     : if False, skip R script if CSV already exists

    Returns:
        pd.DataFrame with all games combined, plus columns:
            game_number, period, split
    """
    os.makedirs(output_folder, exist_ok=True)

    all_dfs = []
    errors  = []

    for filename in sorted(os.listdir(dvw_folder)):
        if not filename.endswith(".dvw"):
            continue

        # parse metadata from filename 
        parts = filename.replace(".dvw", "").split("_")
        if len(parts) < 3:
            print(f"Skipping {filename} — unexpected naming format")
            continue

        game_num = int(parts[0])
        period   = parts[1]    # pre, conf1, conf2
        split    = parts[2]    # train or test

        dvw_path = os.path.join(dvw_folder, filename)
        out_path = os.path.join(output_folder, filename.replace(".dvw", ".csv"))

        # run R script 
        # skip if CSV exists and force_rerun=False
        if force_rerun or not os.path.exists(out_path):
            result = subprocess.run([
                "Rscript", r_script_path,
                dvw_path,
                str(focus_team_id),
                focus_team_name,
                out_path
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"ERROR on {filename}:\n{result.stderr}")
                errors.append(filename)
                continue

        # load and tag
        try:
            game_df = pd.read_csv(out_path)
            game_df['game_number'] = game_num
            game_df['period']      = period
            game_df['split']       = split
            all_dfs.append(game_df)
            print(f"  Loaded {filename}: {len(game_df)} rows")
        except Exception as e:
            print(f"ERROR reading {out_path}: {e}")
            errors.append(filename)

    if errors:
        print(f"\nFailed files: {errors}")

    if not all_dfs:
        raise ValueError("No files loaded — check dvw_folder path and filename format")

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows: {len(df_all)}")
    print(df_all.groupby(['period', 'split'])['game_number'].nunique()
              .rename("games").to_string())

    return df_all


# 2. FEATURE ENGINEERING

# setter position (DataVolley) -> team rotation (American convention)
SETTER_TO_ROTATION = {1: 1, 6: 2, 5: 3, 4: 4, 3: 5, 2: 6}

PASS_QUALITY_MAP = {
    '#': 4,
    '+': 3,
    '!': 2,
    '-': 1,
    '/': 0.5,
    '=': 0
}

def build_features(df, focus_team, le_server=None, le_player=None, le_prev=None):
    """
    Apply all feature engineering to a DataFrame of serve-receive plays.
    
    Handles:
        - Filtering to focus team receiving
        - Pass quality mapping
        - Rotation derivation
        - Role-aware player encoding (recv/opp front and back row)
        - Lag features (prev_point_won, prev_attacker)

    Parameters:
        df          : raw combined DataFrame from load_and_combine_games()
        focus_team  : team name string to filter receiving plays
        le_server   : fitted LabelEncoder for server_id (pass fitted encoder
                      from training set to apply same encoding to test)
        le_player   : fitted LabelEncoder for player IDs
        le_prev     : fitted LabelEncoder for prev_attacker_id

    Returns:
        df_focus    : feature-engineered DataFrame
        encoders    : dict of fitted encoders {server, player, prev}
                      (use these to encode test/val sets consistently)
    """
    # filter to focus team receiving
    df_focus = df[
        (df['receiving_team'] == focus_team) &
        (df['no_attack'] == False)
    ].copy()

    print(f"Rows where {focus_team} is receiving (with attack): {len(df_focus)}")

    # pass quality
    df_focus['pass_quality'] = df_focus['receive_eval_code'].map(PASS_QUALITY_MAP)
    unmapped = df_focus['pass_quality'].isna().sum()
    if unmapped > 0:
        print(f"Warning: {unmapped} unmapped receive codes")

    # receiving rotation
    setter_pos = np.where(
        df_focus['receiving_team'] == df_focus['home_team'],
        df_focus['home_setter_position'],
        df_focus['visiting_setter_position']
    )
    df_focus['receiving_rotation'] = (
        pd.Series(setter_pos, index=df_focus.index).map(SETTER_TO_ROTATION)
    )

    # role-aware player columns
    for pos in range(1, 7):
        home_col     = f'home_player_id{pos}'
        visiting_col = f'visiting_player_id{pos}'
        df_focus[f'recv_p{pos}'] = np.where(
            df_focus['receiving_team'] == df_focus['home_team'],
            df_focus[home_col], df_focus[visiting_col]
        )
        df_focus[f'opp_p{pos}'] = np.where(
            df_focus['serving_team'] == df_focus['home_team'],
            df_focus[home_col], df_focus[visiting_col]
        )

    # encode server
    if le_server is None:
        le_server = LabelEncoder()
        le_server.fit(df_focus['server_id'].astype(str))
    df_focus['server_encoded'] = df_focus['server_id'].astype(str).apply(
        lambda x: le_server.transform([x])[0] if x in le_server.classes_ else -1
    )

    # encode all player columns
    player_cols = [f'recv_p{p}' for p in range(1, 7)] + \
                  [f'opp_p{p}'  for p in range(1, 7)]

    if le_player is None:
        le_player = LabelEncoder()
        le_player.fit(df_focus[player_cols].values.flatten().astype(str))

    # handle unsee players safely
    def safe_player_encode(val, le):
        val = str(val)
        return le.transform([val])[0] if val in le.classes_ else -1

    for col in player_cols:
        df_focus[col + '_enc'] = df_focus[col].astype(str).apply(
            lambda x: safe_player_encode(x, le_player)
        )

    # lag features
    df_focus = df_focus.sort_values(
        ['game_number', 'set_number', 'point_id']
    ).reset_index(drop=True)

    # detect consecutive serve-receive points:
    # point_id exactly 1 greater than previous row in same set and game
    # means they did not side out — they stayed in serve receive
    df_focus['prev_point_id'] = df_focus['point_id'].shift(1)
    df_focus['prev_set']      = df_focus['set_number'].shift(1)
    df_focus['prev_game']     = df_focus['game_number'].shift(1)

    df_focus['is_consecutive_sr'] = (
        (df_focus['point_id']    == df_focus['prev_point_id'] + 1) &
        (df_focus['set_number']  == df_focus['prev_set']) &
        (df_focus['game_number'] == df_focus['prev_game'])
    )

    # prev_point_won: only meaningful if consecutive serve-receive point
    # otherwise 0.0 (neutral — no streak info)
    df_focus['prev_attacker_id'] = df_focus['attacker_id'].shift(1)
    df_focus['prev_point_won']   = np.where(
        df_focus['is_consecutive_sr'],
        (df_focus['point_won_by'].shift(1) == focus_team).astype(float),
        0.0
    )

    # prev_attacker_enc: only valid if consecutive serve-receive point
    # otherwise 'none' (not applicable)
    df_focus['prev_attacker_id_clean'] = np.where(
        df_focus['is_consecutive_sr'],
        df_focus['prev_attacker_id'].fillna('none').astype(str),
        'none'
    )

    if le_prev is None:
        le_prev = LabelEncoder()
        le_prev.fit(df_focus['prev_attacker_id_clean'].astype(str))

    df_focus['prev_attacker_enc'] = df_focus['prev_attacker_id_clean'].apply(
        lambda x: le_prev.transform([x])[0] if x in le_prev.classes_ else -1
    )

    # drop helper columns
    df_focus = df_focus.drop(columns=[
        'prev_point_id', 'prev_set', 'prev_game',
        'is_consecutive_sr', 'prev_attacker_id', 'prev_attacker_id_clean'
    ])

    encoders = {
        'server': le_server,
        'player': le_player,
        'prev':   le_prev
    }

    return df_focus, encoders


def add_attack_profiles(df_focus, train_df):
    """
    Add player_top_attack_enc feature derived from training data only.
    Must be called after train/test split.

    Parameters:
        df_focus : full feature-engineered DataFrame
        train_df : training subset of df_focus

    Returns:
        df_focus with player_top_attack_enc column added
        le_attack : fitted LabelEncoder for attack codes
    """
    attack_profile = (
        train_df.groupby(['attacker_id', 'attack_code'])
        .size()
        .reset_index(name='count')
    )
    attack_profile['attack_code_pct'] = (
        attack_profile.groupby('attacker_id')['count']
        .transform(lambda x: x / x.sum())
    )
    top_attack = (
        attack_profile.sort_values('attack_code_pct', ascending=False)
        .groupby('attacker_id').first()
        .reset_index()
        .rename(columns={'attack_code': 'player_top_attack_code'})
        [['attacker_id', 'player_top_attack_code']]
    )

    le_attack = LabelEncoder()
    le_attack.fit(top_attack['player_top_attack_code'].astype(str))
    top_attack['player_top_attack_enc'] = le_attack.transform(
        top_attack['player_top_attack_code'].astype(str)
    )

    df_focus['player_top_attack_enc'] = df_focus['attacker_id'].map(
        top_attack.set_index('attacker_id')['player_top_attack_enc']
    )

    return df_focus, le_attack


def encode_target(df_focus, train_df):
    """
    Fit LabelEncoder on training attackers only and encode target column.
    Attackers unseen in training get NaN (dropped downstream).

    Parameters:
        df_focus : full feature-engineered DataFrame
        train_df : training subset

    Returns:
        df_focus with target column added
        le       : fitted LabelEncoder
    """
    le = LabelEncoder()
    le.fit(train_df['attacker_id'].apply(
        lambda x: str(int(float(x)))
    ).astype(str))

    def safe_encode(val):
        val = str(int(float(val)))
        return le.transform([val])[0] if val in le.classes_ else np.nan

    df_focus['target'] = df_focus['attacker_id'].apply(safe_encode)

    # report unseen attackers
    all_attackers   = set(df_focus['attacker_id'].apply(
        lambda x: str(int(float(x)))
    ))
    train_attackers = set(le.classes_)
    unseen = all_attackers - train_attackers
    if unseen:
        print(f"Note: {len(unseen)} attacker(s) unseen in training: {unseen}")

    return df_focus, le


# 3. PREDICTION HELPERS

def mask_probabilities(proba_row, valid_ids, all_classes):
    """
    Zero out probabilities for players not on court, renormalize to sum to 1.

    Parameters:
        proba_row   : np.array of raw model probabilities
        valid_ids   : set of player ID strings valid for this point
        all_classes : le.classes_ from the fitted LabelEncoder

    Returns:
        np.array of masked, renormalized probabilities
    """
    masked = proba_row.copy()
    for i, cls in enumerate(all_classes[:len(proba_row)]):
        if cls not in valid_ids:
            masked[i] = 0.0
    total = masked.sum()
    return masked / total if total > 0 else masked


def get_masked_probas(df_focus, model, features, le):
    """
    Generate masked probability arrays for all rows in df_focus.

    Parameters:
        df_focus : DataFrame with recv_p1-6 columns
        model    : fitted XGBClassifier
        features : list of feature column names
        le       : fitted LabelEncoder

    Returns:
        proba_df : DataFrame of masked probabilities (rows=points, cols=players)
    """
    all_masked = []
    for idx in df_focus.index:
        row   = df_focus.loc[idx]
        valid = {
            str(int(float(row[f'recv_p{p}'])))
            for p in range(1, 7)
            if pd.notna(row[f'recv_p{p}'])
        }
        raw = model.predict_proba(df_focus.loc[[idx], features])[0]
        all_masked.append(mask_probabilities(raw, valid, le.classes_))

    return pd.DataFrame(
        all_masked,
        columns=le.classes_,
        index=df_focus.index
    ).astype(float)


# 4. PREDICTABILITY ANALYSIS

def compute_predictability(df_focus, proba_df, le, focus_team):
    """
    Compute predictability metrics and add them to df_focus.

    Metrics added:
        pred_entropy        : Shannon entropy of probability distribution
        top_predicted       : player model predicted most likely
        prediction_correct  : did actual attacker match top prediction?
        chose_non_top       : did setter choose someone other than top pick?

    Parameters:
        df_focus : feature-engineered DataFrame (index must match proba_df)
        proba_df : masked probability DataFrame from get_masked_probas()
        le       : fitted LabelEncoder
        focus_team : team name string (for display)

    Returns:
        df_focus with predictability columns added
        summary  : dict of key metrics
    """
    df_focus = df_focus.copy()

    # standardize attacker_id for comparison
    df_focus['attacker_id_str'] = df_focus['attacker_id'].apply(
        lambda x: str(int(float(x)))
    )

    df_focus['pred_entropy'] = proba_df.apply(
        lambda row: shannon_entropy(row[row > 0]), axis=1
    )

    df_focus['top_predicted'] = le.classes_[
        np.argmax(proba_df.values, axis=1)
    ]

    df_focus['prediction_correct'] = (
        df_focus['top_predicted'] == df_focus['attacker_id_str']
    )

    df_focus['chose_non_top'] = (
        df_focus['top_predicted'] != df_focus['attacker_id_str']
    )

    # summary
    df_in = df_focus[df_focus['pass_quality'] >= 3]

    summary = {
        'overall_accuracy':      df_focus['prediction_correct'].mean(),
        'insystem_accuracy':     df_in['prediction_correct'].mean(),
        'non_top_rate_all':      df_focus['chose_non_top'].mean(),
        'non_top_rate_insystem': df_in['chose_non_top'].mean(),
        'mean_entropy_all':      df_focus['pred_entropy'].mean(),
        'mean_entropy_insystem': df_in['pred_entropy'].mean(),
        'entropy_by_rotation':   df_focus.groupby('receiving_rotation')['pred_entropy'].mean(),
        'non_top_by_rotation':   df_focus.groupby('receiving_rotation')['chose_non_top'].mean(),
    }

    return df_focus, summary


def print_predictability_summary(summary, label=""):
    """Pretty-print the summary dict from compute_predictability()."""
    header = f"=== Predictability Summary {label} ==="
    print(header)
    print()
    print(f"  Prediction accuracy (all passes):      {summary['overall_accuracy']:.1%}")
    print(f"  Prediction accuracy (in-system 3-4):   {summary['insystem_accuracy']:.1%}")
    print()
    print(f"  Non-top choice rate (all passes):      {summary['non_top_rate_all']:.1%}")
    print(f"  Non-top choice rate (in-system 3-4):   {summary['non_top_rate_insystem']:.1%}")
    print()
    print(f"  Mean entropy (all):                    {summary['mean_entropy_all']:.3f}")
    print(f"  Mean entropy (in-system):              {summary['mean_entropy_insystem']:.3f}")
    print()
    print("  Entropy by rotation (higher = less predictable):")
    print(summary['entropy_by_rotation'].sort_values().to_string())
    print()
    print("  Non-top choice rate by rotation:")
    print(summary['non_top_by_rotation'].round(3).to_string())

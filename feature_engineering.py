import os, json, sqlite3, numpy as np, pandas as pd; from collections import defaultdict
from config import ROLES, FEATURE_CACHE_PATH, CHAMPIONS_PATH, DB_PATH

def build_feature_matrices(df_train):
    # 1. Collect match magic vs physical damage for each champion
    champ_match_ap = defaultdict(list)
    for side in ['blue', 'red']:
        for role in ROLES:
            phys_col = f'{side}_{role}_phys'
            magic_col = f'{side}_{role}_magic'
            if phys_col in df_train.columns and magic_col in df_train.columns:
                for champ, phys, magic in zip(df_train[f'{side}_{role}'], df_train[phys_col], df_train[magic_col]):
                    total_dmg = float(phys or 0) + float(magic or 0)
                    if champ and total_dmg > 0:
                        champ_match_ap[champ].append(float(magic or 0) / total_dmg)

    # 2. Compute mean AP damage ratio and damage variance per champion
    ap_ratios = {}
    ap_variances = {}
    for champ, ratios in champ_match_ap.items():
        if ratios:
            ap_ratios[champ] = float(np.mean(ratios))
            ap_variances[champ] = float(np.var(ratios)) if len(ratios) >= 5 else 0.0

    # 3. Track role pick counts and 1v1 lane matchup wins
    role_counts = defaultdict(lambda: [0, 0, 0, 0, 0])
    counter_stats = defaultdict(lambda: [0, 0])

    winning_teams = df_train['winning_team'].to_numpy()
    for side in ['blue', 'red']:
        for role_idx, role in enumerate(ROLES):
            champs_in_role = df_train[f'{side}_{role}'].to_numpy()
            for champ in champs_in_role:
                if champ:
                    role_counts[champ][role_idx] += 1

    # 4. Count 1v1 lane matchup wins for Top, Mid, and Support
    for role in ['top', 'mid', 'support']:
        blue_champs = df_train[f'blue_{role}'].to_numpy()
        red_champs = df_train[f'red_{role}'].to_numpy()
        blue_win = (winning_teams == 'BLUE_WIN')
        for blue_champ, red_champ, blue_won in zip(blue_champs, red_champs, blue_win):
            if blue_champ and red_champ:
                key_blue = f"{role}:{blue_champ}_vs_{red_champ}"
                counter_stats[key_blue][0] += int(blue_won)
                counter_stats[key_blue][1] += 1

                key_red = f"{role}:{red_champ}_vs_{blue_champ}"
                counter_stats[key_red][0] += int(not blue_won)
                counter_stats[key_red][1] += 1

    # 5. Load champion list and compute role play frequencies
    all_champs = []
    if os.path.exists(CHAMPIONS_PATH):
        with open(CHAMPIONS_PATH, 'r') as file:
            all_champs = json.load(file)
    else:
        all_champs = list(role_counts.keys())

    role_freqs = {}
    for champ in all_champs:
        if champ in role_counts and sum(role_counts[champ]) >= 10:
            counts = role_counts[champ]
            total_games = sum(counts)
            role_freqs[champ] = {role: float(counts[i] / total_games) for i, role in enumerate(ROLES)}
        else:
            role_freqs[champ] = {role: 0.2 for role in ROLES}

    # 6. Compute Empirical Bayes shrinkage 1v1 counter advantage matrix
    counter_mat = {}
    for key, (wins, games) in counter_stats.items():
        if games >= 10:
            counter_mat[key] = ((wins / games) - 0.5) * (games / (games + 30.0))

    # 7. Cache feature matrices to disk
    data = {
        'champ_ap_ratios': ap_ratios,
        'champ_ap_variances': ap_variances,
        'role_freqs': role_freqs,
        'counter_matrix': counter_mat
    }
    with open(FEATURE_CACHE_PATH, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Cached feature matrices to {FEATURE_CACHE_PATH}")
    return data

def load_feature_matrices(df_train=None):
    """Load cached feature matrices from JSON or rebuild if missing."""
    if df_train is None and os.path.exists(FEATURE_CACHE_PATH):
        with open(FEATURE_CACHE_PATH, 'r') as file:
            return json.load(file)
    return build_feature_matrices(df_train)

def extract_draft_features(blue_champs, red_champs, feature_matrices=None):
    """Extract 44 continuous domain features from a 10-champion draft vector."""
    if feature_matrices is None:
        feature_matrices = load_feature_matrices()

    ap_ratio_dict = feature_matrices.get('champ_ap_ratios', {})
    ap_variance_dict = feature_matrices.get('champ_ap_variances', {})
    role_freq_dict = feature_matrices.get('role_freqs', {})
    counter_stats_dict = feature_matrices.get('counter_matrix', {})

    # 1. Slot Features: 40 continuous features (20 Blue + 20 Red)
    slot_features = []
    for side_champs in [blue_champs, red_champs]:
        for i, champ in enumerate(side_champs):
            role = ROLES[i]
            if champ:
                slot_rf = float(role_freq_dict.get(champ, {}).get(role, 0.0))
                slot_ap = float(ap_ratio_dict.get(champ, 0.5))
                slot_var = float(ap_variance_dict.get(champ, 0.0))
                is_off_meta = max(0.0, 0.2 - slot_rf)
                slot_features.extend([slot_ap, slot_var, slot_rf, is_off_meta])
            else:
                slot_features.extend([0.0, 0.0, 0.0, 0.0])

    # 2. Lane Matchup Counters: 3 features (Top, Mid, Support)
    lane_counters = []
    for role_idx in [0, 2, 4]:
        role = ROLES[role_idx]
        champ_blue, champ_red = blue_champs[role_idx], red_champs[role_idx]
        if champ_blue and champ_red:
            blue_counter_val = float(counter_stats_dict.get(f"{role}:{champ_blue}_vs_{champ_red}", 0.0))
            red_counter_val = float(counter_stats_dict.get(f"{role}:{champ_red}_vs_{champ_blue}", 0.0))
            lane_counters.append(blue_counter_val - red_counter_val)
        else:
            lane_counters.append(0.0)

    # 3. Team AP Balance Delta: 1 feature (Blue total AP - Red total AP)
    blue_ap_total = sum(slot_features[i * 4] for i in range(5))
    red_ap_total = sum(slot_features[(5 + i) * 4] for i in range(5))
    ap_balance_delta = blue_ap_total - red_ap_total

    # Combine into 44D feature vector
    return np.array(slot_features + lane_counters + [ap_balance_delta], dtype=np.float32)

if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        from sklearn.model_selection import train_test_split
        conn = sqlite3.connect(DB_PATH)
        df_all = pd.read_sql_query(
            "SELECT * FROM matches WHERE game_version LIKE ?",
            conn,
            params=("16%",)
        )
        conn.close()
        df_tr, _ = train_test_split(df_all, test_size=0.2, random_state=42)
        build_feature_matrices(df_tr)

import os, json, sqlite3, numpy as np, pandas as pd; from collections import defaultdict
from config import ROLES, FEATURE_CACHE_PATH, CHAMPIONS_PATH, DB_PATH

def build_feature_matrices(df_train):
    champ_match_ap = defaultdict(list)
    for side in ['blue', 'red']:
        for r in ROLES:
            p_col = f'{side}_{r}_phys' if f'{side}_{r}_phys' in df_train.columns else f'{side}_supp_phys'
            m_col = f'{side}_{r}_magic' if f'{side}_{r}_magic' in df_train.columns else f'{side}_supp_magic'
            if p_col in df_train.columns and m_col in df_train.columns:
                for c, p, m in zip(df_train[f'{side}_{r}'], df_train[p_col], df_train[m_col]):
                    if c:
                        tot = float(p or 0) + float(m or 0)
                        if tot > 0:
                            champ_match_ap[c].append(float(m or 0) / tot)

    ap_ratios = {c: float(np.mean(ratios)) for c, ratios in champ_match_ap.items() if ratios}
    ap_variances = {c: float(np.var(ratios)) if len(ratios) >= 5 else 0.0 for c, ratios in champ_match_ap.items() if ratios}

    role_counts = defaultdict(lambda: [0, 0, 0, 0, 0])
    counter_stats = defaultdict(lambda: [0, 0])

    winning_teams = df_train['winning_team'].to_numpy()
    for side in ['blue', 'red']:
        for r_idx, r in enumerate(ROLES):
            col = df_train[f'{side}_{r}'].to_numpy()
            for c in col:
                if c:
                    role_counts[c][r_idx] += 1

    for r in ['top', 'mid', 'support']:
        b_col, r_col = df_train[f'blue_{r}'].to_numpy(), df_train[f'red_{r}'].to_numpy()
        b_wins = (winning_teams == 'BLUE_WIN')
        for b_c, r_c, b_win in zip(b_col, r_col, b_wins):
            if b_c and r_c:
                k_b = f"{r}:{b_c}_vs_{r_c}"
                counter_stats[k_b][0] += int(b_win)
                counter_stats[k_b][1] += 1
                k_r = f"{r}:{r_c}_vs_{b_c}"
                counter_stats[k_r][0] += int(not b_win)
                counter_stats[k_r][1] += 1

    all_champs = []
    if os.path.exists(CHAMPIONS_PATH):
        with open(CHAMPIONS_PATH, 'r') as f:
            all_champs = json.load(f)
    else:
        all_champs = list(role_counts.keys())

    role_freqs = {}
    for c in all_champs:
        if c in role_counts and sum(role_counts[c]) >= 10:
            r_list = role_counts[c]
            tot = sum(r_list)
            role_freqs[c] = {r: float(r_list[i] / tot) for i, r in enumerate(ROLES)}
        else:
            role_freqs[c] = {r: 0.2 for r in ROLES}

    counter_mat = {}
    for k, (w, g) in counter_stats.items():
        if g >= 10:
            counter_mat[k] = ((w / g) - 0.5) * (g / (g + 30.0))

    data = {
        'champ_ap_ratios': ap_ratios,
        'champ_ap_variances': ap_variances,
        'role_freqs': role_freqs,
        'counter_matrix': counter_mat
    }
    with open(FEATURE_CACHE_PATH, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Cached feature matrices to {FEATURE_CACHE_PATH}")
    return data

def load_feature_matrices(df_train=None):
    if df_train is None and os.path.exists(FEATURE_CACHE_PATH):
        with open(FEATURE_CACHE_PATH, 'r') as f:
            return json.load(f)
    return build_feature_matrices(df_train)

def extract_draft_features(blue_champs, red_champs, feature_matrices=None):
    if feature_matrices is None:
        feature_matrices = load_feature_matrices()

    ap_ratio_dict = feature_matrices.get('champ_ap_ratios', {})
    ap_variance_dict = feature_matrices.get('champ_ap_variances', {})
    role_freq_dict = feature_matrices.get('role_freqs', {})
    counter_stats_dict = feature_matrices.get('counter_matrix', {})

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

    lane_counters = []
    for role_idx in [0, 2, 4]:  # Top, Mid, Support (1v1 lane counter roles)
        role = ROLES[role_idx]
        champ_blue, champ_red = blue_champs[role_idx], red_champs[role_idx]
        if champ_blue and champ_red:
            blue_counter_val = float(counter_stats_dict.get(f"{role}:{champ_blue}_vs_{champ_red}", 0.0))
            red_counter_val = float(counter_stats_dict.get(f"{role}:{champ_red}_vs_{champ_blue}", 0.0))
            lane_counters.append(blue_counter_val - red_counter_val)
        else:
            lane_counters.append(0.0)

    blue_ap_total = sum(slot_features[i * 4] for i in range(5))
    red_ap_total = sum(slot_features[(5 + i) * 4] for i in range(5))
    ap_balance_delta = blue_ap_total - red_ap_total

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

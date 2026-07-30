import os, json, sqlite3, numpy as np, pandas as pd
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, 'feature_matrices.json')
ROLES = ['top', 'jungle', 'mid', 'bot', 'support']
ROLE_IDX = {r: i for i, r in enumerate(ROLES)}
DUO_WEIGHTS = {('bot', 'support'): 2.0, ('mid', 'jungle'): 2.0, ('mid', 'bot'): 1.5}

NUM_SLOT_FEATS_PER_CHAMP = 3
NUM_SLOT_TOTAL_FEATS = 10 * NUM_SLOT_FEATS_PER_CHAMP  # 30 per-slot features

def build_feature_matrices(df_train):
    """
    Compute historical domain matrices from training split ONLY (prevents val data leakage):
    1. Empirical Champion AP / AD Damage Ratios & Variance (Flex Builder detection)
    2. Pairwise Duo Synergy Win Rates (Bot+Supp, Mid+Jgl, Mid+Bot)
    3. 1v1 Lane Matchup Counter Win Rates
    """
    if df_train is None or len(df_train) == 0:
        raise ValueError("df_train must be provided explicitly from training split to prevent validation data leakage!")

    champ_match_ap = defaultdict(list)
    if 'blue_top_phys' in df_train.columns:
        for side in ['blue', 'red']:
            for r in ROLES:
                for c, p, m in zip(df_train[f'{side}_{r}'], df_train[f'{side}_{r}_phys'], df_train[f'{side}_{r}_magic']):
                    if c:
                        tot = float(p or 0) + float(m or 0)
                        if tot > 0:
                            champ_match_ap[c].append(float(m or 0) / tot)

    ap_ratios = {}
    ap_variances = {}
    for c, ratios in champ_match_ap.items():
        if ratios:
            ap_ratios[c] = float(np.mean(ratios))
            ap_variances[c] = float(np.var(ratios)) if len(ratios) >= 5 else 0.0
        else:
            ap_ratios[c] = 0.5
            ap_variances[c] = 0.0

    # Track individual champion role frequencies (e.g. Ahri Mid = 0.985, Ahri Jungle = 0.001)
    role_counts = defaultdict(lambda: [0, 0, 0, 0, 0])
    champ_stats = defaultdict(lambda: [0, 0])
    synergy_stats, counter_stats = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])

    for _, row in df_train.iterrows():
        b_win = 1 if row['winning_team'] == 'BLUE_WIN' else 0

        for side, win in [('blue', b_win), ('red', 1 - b_win)]:
            for r_idx, r in enumerate(ROLES):
                c = row[f'{side}_{r}']
                if c:
                    champ_stats[c][0] += win
                    champ_stats[c][1] += 1
                    role_counts[c][r_idx] += 1

            for (r1, r2) in DUO_WEIGHTS.keys():
                c1, c2 = row[f'{side}_{r1}'], row[f'{side}_{r2}']
                if c1 and c2:
                    k = f"{r1}:{c1}|{r2}:{c2}"
                    synergy_stats[k][0] += win
                    synergy_stats[k][1] += 1

        # Process 1v1 lane counters
        for r in ROLES:
            b_c, r_c = row[f'blue_{r}'], row[f'red_{r}']
            if b_c and r_c:
                k = f"{r}:{b_c}_vs_{r_c}"
                counter_stats[k][0] += b_win
                counter_stats[k][1] += 1

    champ_winrates = {c: w / g for c, (w, g) in champ_stats.items() if g >= 5}

    # Role frequency matrix per champion with minimum 10 games sample threshold
    role_freqs = {}
    for c, r_list in role_counts.items():
        tot = sum(r_list)
        if tot >= 10:
            role_freqs[c] = {r: float(r_list[i] / tot) for i, r in enumerate(ROLES)}
        else:
            role_freqs[c] = {r: 0.2 for r in ROLES}

    # True Residual Synergy Lift with Bayesian Empirical Shrinkage (g / (g + 30.0))
    synergy_mat = {}
    for k, (w, g) in synergy_stats.items():
        if g >= 10:
            r1_part, r2_part = k.split('|')
            c1 = r1_part.split(':', 1)[1]
            c2 = r2_part.split(':', 1)[1]
            base_wr = max(champ_winrates.get(c1, 0.5), champ_winrates.get(c2, 0.5))
            shrinkage = g / (g + 30.0)
            synergy_mat[k] = ((w / g) - base_wr) * shrinkage

    # True Residual Counter Lift with Bayesian Empirical Shrinkage (g / (g + 30.0))
    counter_mat = {}
    for k, (w, g) in counter_stats.items():
        if g >= 10:
            role_part, champs_part = k.split(':', 1)
            b_c, r_c = champs_part.split('_vs_', 1)
            base_wr = champ_winrates.get(b_c, 0.5)
            shrinkage = g / (g + 30.0)
            counter_mat[k] = ((w / g) - base_wr) * shrinkage

    data = {
        'champ_ap_ratios': ap_ratios,
        'champ_ap_variances': ap_variances,
        'champ_winrates': champ_winrates,
        'role_freqs': role_freqs,
        'synergy_matrix': synergy_mat,
        'counter_matrix': counter_mat
    }
    with open(CACHE_PATH, 'w') as f: json.dump(data, f)
    print(f"Cached leakage-free feature matrices from {len(df_train)} training matches to {CACHE_PATH}")
    return data

def load_feature_matrices(df_train=None):
    if df_train is None and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r') as f: return json.load(f)
    return build_feature_matrices(df_train)

def extract_draft_features(blue_champs, red_champs, mat=None):
    if mat is None: mat = load_feature_matrices()
    ap_map = mat.get('champ_ap_ratios', {})
    var_map = mat.get('champ_ap_variances', {})
    rf_map = mat.get('role_freqs', {})
    syn_map = mat.get('synergy_matrix', {})
    cnt_map = mat.get('counter_matrix', {})

    # 1. Primary Per-Slot Champion Features (10 slots x 3 features = 30 floats)
    slot_features = []
    for side_champs in [blue_champs, red_champs]:
        for i in range(5):
            c = side_champs[i] if i < len(side_champs) else ''
            r = ROLES[i]
            if c:
                slot_ap = float(ap_map.get(c, 0.5))
                slot_var = float(var_map.get(c, 0.0))
                slot_rf = float(rf_map.get(c, {}).get(r, 0.2))
                slot_features.extend([slot_ap, slot_var, slot_rf])
            else:
                slot_features.extend([0.0, 0.0, 0.0])

    # 2. Solo-Lane 1v1 True Residual Counter Matchups (Top and Mid)
    b_cnt, r_cnt = 0.0, 0.0
    for r in ['top', 'mid']:
        i = ROLE_IDX[r]
        c_b = blue_champs[i] if i < len(blue_champs) else ''
        c_r = red_champs[i] if i < len(red_champs) else ''
        if c_b and c_r:
            b_cnt += float(cnt_map.get(f"{r}:{c_b}_vs_{c_r}", 0.0))
            r_cnt += float(cnt_map.get(f"{r}:{c_r}_vs_{c_b}", 0.0))

    net_synergy = 0.0
    net_counter = float(b_cnt - r_cnt)
    return np.array(slot_features + [net_synergy, net_counter], dtype=np.float32)

if __name__ == "__main__":
    db_path = os.path.join(SCRIPT_DIR, 'league_data.db')
    if os.path.exists(db_path):
        from sklearn.model_selection import train_test_split
        conn = sqlite3.connect(db_path)
        df_all = pd.read_sql_query("SELECT * FROM matches", conn)
        conn.close()
        df_tr, _ = train_test_split(df_all, test_size=0.2, random_state=42)
        build_feature_matrices(df_tr)

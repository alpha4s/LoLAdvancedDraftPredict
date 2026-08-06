import os, sys, json, torch, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config import MODEL_META_PATH, MODEL_WEIGHTS_PATH, ROLES, EMBEDDING_DIM
from model import WideAndDeepDraftNN
from feature_engineering import load_feature_matrices

class DraftPredictor:
    def __init__(self):
        with open(MODEL_META_PATH, 'r') as f:
            meta = json.load(f)

        self.champion_names = meta['champion_names']
        self.champ_to_idx = meta['champ_to_idx']
        self.num_champs = len(self.champion_names)
        self.roles = ROLES

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build feature tensors for model lookup
        feature_matrices = load_feature_matrices()
        zeros = lambda *shape: np.zeros(shape, dtype=np.float32)
        get_mat = lambda key: feature_matrices.get(key, {})
        to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32).to(self.device)

        stride = self.num_champs + 1
        ap_ratios = zeros(self.num_champs + 1, 1)
        ap_variances = zeros(self.num_champs + 1, 1)
        role_freqs = zeros(self.num_champs + 1, 5)

        top_counters = zeros(stride**2, 1)
        mid_counters = zeros(stride**2, 1)
        supp_counters = zeros(stride**2, 1)

        ap_ratio_dict = get_mat('champ_ap_ratios')
        ap_variance_dict = get_mat('champ_ap_variances')
        role_freq_dict = get_mat('role_freqs')
        counter_stats_dict = get_mat('counter_matrix')

        for champ, idx in self.champ_to_idx.items():
            ap_ratios[idx, 0] = float(ap_ratio_dict.get(champ, .5))
            ap_variances[idx, 0] = float(ap_variance_dict.get(champ, 0))
            for role_idx, role in enumerate(self.roles):
                role_freqs[idx, role_idx] = float(role_freq_dict.get(champ, {}).get(role, 0))

        for champ_blue, idx_blue in self.champ_to_idx.items():
            for champ_red, idx_red in self.champ_to_idx.items():
                flat_idx = idx_blue * stride + idx_red
                top_counters[flat_idx, 0] = float(counter_stats_dict.get(f"top:{champ_blue}_vs_{champ_red}", 0))
                mid_counters[flat_idx, 0] = float(counter_stats_dict.get(f"mid:{champ_blue}_vs_{champ_red}", 0))
                supp_counters[flat_idx, 0] = float(counter_stats_dict.get(f"support:{champ_blue}_vs_{champ_red}", 0))

        feature_tensors = {
            'ap_ratios': to_tensor(ap_ratios),
            'ap_variances': to_tensor(ap_variances),
            'role_freqs': to_tensor(role_freqs),
            'top_counters': to_tensor(top_counters),
            'mid_counters': to_tensor(mid_counters),
            'supp_counters': to_tensor(supp_counters)
        }

        self.model = WideAndDeepDraftNN(
            num_champs=self.num_champs,
            features=feature_tensors,
            embed_dim=meta.get('embedding_dim', EMBEDDING_DIM)
        ).to(self.device)

        if os.path.exists(MODEL_WEIGHTS_PATH):
            self.model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=self.device))
        self.model.eval()

    def process_team(self, team):
        names = [team.get(r, '') for r in self.roles]
        idxs = [self.champ_to_idx.get(c, self.num_champs) for c in names]
        return names, idxs

    def run_inference(self, deep_indices):
        d_tensor = torch.tensor(deep_indices, dtype=torch.long, device=self.device)
        with torch.no_grad():
            return self.model(d_tensor).squeeze(-1).cpu().numpy()

    def predict(self, blue_team, red_team):
        b_team = {k.lower(): v for k, v in blue_team.items()}
        r_team = {k.lower(): v for k, v in red_team.items()}

        _, b_idxs = self.process_team(b_team)
        _, r_idxs = self.process_team(r_team)

        prob = float(self.run_inference([b_idxs + r_idxs])[0])
        return {
            "probability": prob,
            "blue_roster": {r: b_team.get(r) or "Empty" for r in self.roles},
            "red_roster": {r: r_team.get(r) or "Empty" for r in self.roles}
        }

    def recommend(self, blue_team, red_team, user_side, user_role, candidates):
        user_side, user_role = user_side.lower(), user_role.lower()
        target_idx = self.roles.index(user_role)
        _, b_idxs = self.process_team({k.lower(): v for k, v in blue_team.items()})
        _, r_idxs = self.process_team({k.lower(): v for k, v in red_team.items()})

        base_deep = b_idxs + r_idxs
        slot = target_idx if user_side == 'blue' else 5 + target_idx

        valid_candidates = [c for c in candidates if c in self.champ_to_idx]
        if not valid_candidates:
            return {"recommendations": []}

        num_cands = len(valid_candidates)
        batch_deep = np.tile(base_deep, (num_cands, 1))

        for i, cand in enumerate(valid_candidates):
            batch_deep[i, slot] = self.champ_to_idx[cand]

        probs = self.run_inference(batch_deep)
        baseline = float(self.run_inference([base_deep])[0])
        user_baseline = baseline if user_side == 'blue' else 1.0 - baseline

        recs = [
            {"name": cand, "win_rate": float(probs[i]) if user_side == 'blue' else 1.0 - float(probs[i])}
            for i, cand in enumerate(valid_candidates)
        ]
        recs.sort(key=lambda x: x['win_rate'], reverse=True)
        return {"baseline": user_baseline, "recommendations": recs[:10]}

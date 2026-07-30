import os
import sys
import json
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from model import WideAndDeepDraftNN
from feature_engineering import load_feature_matrices, extract_draft_features

def get_champion_by_name(name, champ_to_idx):
    """
    Search mapping dictionary for champion name using alphanumeric normalization and substring fallback.
    """
    if not name:
        return None

    norm_input = "".join(c for c in str(name).lower() if c.isalnum())
    if not norm_input:
        return None

    champ_to_idx_norm = {
        "".join(c for c in k.lower() if c.isalnum()): v 
        for k, v in champ_to_idx.items()
    }

    # 1. Exact Normalized Match
    if norm_input in champ_to_idx_norm:
        return champ_to_idx_norm[norm_input]

    # 2. Substring Fallback (for inputs >= 4 chars to avoid short collisions)
    if len(norm_input) >= 4:
        for k, v in champ_to_idx_norm.items():
            if norm_input in k:
                return v

    return None

def predict_match(blue_team, red_team):
    """
    Translate team lineups into model features, load Wide & Deep weights, and output win probabilities.
    """
    model_path = os.path.join(SCRIPT_DIR, 'model_nn.pth')
    meta_path = os.path.join(SCRIPT_DIR, 'model_nn_metadata.json')

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        print("Error: PyTorch Wide & Deep model or metadata not found. Please train the model first.")
        return

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    champion_names = meta['champion_names']
    champ_to_idx = meta['champ_to_idx']
    embedding_dim = meta.get('embedding_dim', 16)
    num_heads = meta.get('num_heads', 2)
    scaler_dict = meta.get('feature_scaler')
    num_champs = len(champion_names)

    if scaler_dict:
        feature_mean = np.array(scaler_dict['mean'], dtype=np.float32)
        feature_std = np.array(scaler_dict['std'], dtype=np.float32)
    else:
        feature_mean, feature_std = np.zeros((1, 32), dtype=np.float32), np.ones((1, 32), dtype=np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideAndDeepDraftNN(num_champs=num_champs, embedding_dim=embedding_dim, num_heads=num_heads, num_extra_features=32)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    blue_team = {k.lower(): v for k, v in blue_team.items()}
    red_team = {k.lower(): v for k, v in red_team.items()}
    roles = ['top', 'jungle', 'mid', 'bot', 'support']

    X_deep = np.full(10, num_champs, dtype=np.int64)

    teams_to_process = [
        ("Blue Team", blue_team, 0),
        ("Red Team", red_team, 5)
    ]

    for side_name, team_dict, offset in teams_to_process:
        print(f"\n{side_name}:")
        for r_idx, role in enumerate(roles):
            champ_name = team_dict.get(role)
            champ_idx = get_champion_by_name(champ_name, champ_to_idx) if champ_name else None

            if champ_idx is not None:
                X_deep[offset + r_idx] = champ_idx
                print(f"  {role.upper():7s}: {champion_names[champ_idx]}")
            else:
                print(f"  {role.upper():7s}: Warning: '{champ_name}' not found in model.")

    blue_champs = [blue_team.get(r, '') for r in roles]
    red_champs = [red_team.get(r, '') for r in roles]
    feature_matrices = load_feature_matrices()
    raw_feats = extract_draft_features(blue_champs, red_champs, feature_matrices).reshape(1, -1)
    norm_feats = (raw_feats - feature_mean) / feature_std

    d_tensor = torch.tensor(X_deep, dtype=torch.long).unsqueeze(0).to(device)
    f_tensor = torch.tensor(norm_feats, dtype=torch.float32).to(device)

    with torch.no_grad():
        win_prob_blue = model(d_tensor, f_tensor).item()

    print(f"\n=================== MATCH PREDICTION ===================")
    print(f"Blue Team Win Probability : {win_prob_blue:.2%}")
    print(f"Red Team Win Probability  : {(1.0 - win_prob_blue):.2%}")
    print(f"Predicted Winner          : {'BLUE WIN' if win_prob_blue > 0.5 else 'RED WIN'}")
    print(f"=======================================================")

if __name__ == "__main__":
    blue_sample = {
        'top': 'Sona',
        'jungle': 'Gwen',
        'mid': 'Soraka',
        'bot': 'Senna',
        'support': 'Taric'
    }
    red_sample = {
        'top': 'Yuumi',
        'jungle': 'Zed',
        'mid': 'Darius',
        'bot': 'Lux',
        'support': 'Malphite'
    }
    predict_match(blue_sample, red_sample)

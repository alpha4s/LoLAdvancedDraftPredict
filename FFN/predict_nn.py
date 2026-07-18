import torch, json, os, numpy as np, torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class WideAndDeepDraftNN(nn.Module):
    """
    Hybrid neural network combining a Wide (linear) model to memorize champion base rates
    and a Deep (embedding + self-attention) model to generalize counter & synergy interactions.
    """
    def __init__(self, num_champs, embedding_dim, num_heads):
        super(WideAndDeepDraftNN, self).__init__()
        self.num_champs = num_champs
        self.embedding_dim = embedding_dim
        
        self.wide_linear = nn.Linear(num_champs * 5, 1, bias=False)
        self.champ_embeddings = nn.Embedding(num_champs + 1, embedding_dim, padding_idx=num_champs)
        self.role_embeddings = nn.Embedding(5, embedding_dim)
        
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True, dropout=0.2)
        self.ln2 = nn.LayerNorm(2 * embedding_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(2 * embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_wide, x_deep):
        wide_out = self.wide_linear(x_wide)
        
        batch_size = x_deep.size(0)
        device = x_deep.device
        champs = self.champ_embeddings(x_deep)
        role_idx = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long, device=device)
        role_idx = role_idx.unsqueeze(0).expand(batch_size, -1)
        roles = self.role_embeddings(role_idx)
        
        seq = champs + roles
        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.ln1(seq + attn_out)
        
        blue_rep = seq[:, :5, :].mean(dim=1)
        red_rep = seq[:, 5:, :].mean(dim=1)
        combined = torch.cat([blue_rep, red_rep], dim=1)
        combined = self.ln2(combined)
        
        deep_out = self.fc(combined)
        return torch.sigmoid(wide_out + deep_out)

def get_champion_by_name(name, champ_to_idx):
    """
    Search mapping dictionary for champion name, resolving common abbreviations and capitalization differences.
    """
    def normalize(s):
        return "".join(c for c in s.lower() if c.isalnum())
    
    aliases = {
        'wukong': 'monkeyking',
        'nunuandwillump': 'nunu',
        'ksante': 'ksante',
        'drmundo': 'drmundo',
        'renataglasc': 'renata',
        'tf': 'twistedfate',
        'mf': 'missfortune',
        'asol': 'aurelionsol',
        'gp': 'gangplank',
        'morg': 'morgana',
        'yi': 'masteryi',
        'blitz': 'blitzcrank',
        'lb': 'leblanc',
        'kass': 'kassadin',
        'mumu': 'amumu',
        'panth': 'pantheon',
        'renek': 'renekton',
        'tahm': 'tahmkench',
        'vlad': 'vladimir',
        'ww': 'warwick',
    }
    
    norm_input = normalize(name)
    if norm_input in aliases:
        norm_input = aliases[norm_input]
        
    champ_to_idx_lower = {k.lower().replace(" ", "").replace("'", ""): v for k, v in champ_to_idx.items()}
    
    if norm_input in champ_to_idx_lower:
        return champ_to_idx_lower[norm_input]
        
    for k, v in champ_to_idx_lower.items():
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
        print("Error: PyTorch Wide & Deep model or metadata not found. Please train/tune the model first.")
        return

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    champion_names = meta['champion_names']
    champ_to_idx = meta['champ_to_idx']
    embedding_dim = meta.get('embedding_dim', 16)
    num_heads = meta.get('num_heads', 1)
    
    num_champs = len(champion_names)
    padding_idx = num_champs
    
    # Initialize the PyTorch model
    model = WideAndDeepDraftNN(num_champs, embedding_dim, num_heads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Lowercase inputs for matching
    blue_team = {k.lower(): v for k, v in blue_team.items()}
    red_team = {k.lower(): v for k, v in red_team.items()}
    
    X_wide = np.zeros(num_champs * 5, dtype=np.float32)
    X_deep = np.zeros(10, dtype=np.int64)
    X_deep.fill(padding_idx)
    
    roles = ['top', 'jungle', 'mid', 'bot', 'support']
    
    print("Blue Team:")
    for r_idx, role in enumerate(roles):
        champ = blue_team.get(role)
        idx = get_champion_by_name(champ, champ_to_idx) if champ else None
        if idx is not None:
            X_deep[r_idx] = idx
            X_wide[r_idx * num_champs + idx] = 1.0
            print(f"  {role.upper():7s}: {champion_names[idx]}")
        else:
            print(f"  {role.upper():7s}: Warning: '{champ}' not found in model.")
    
    print("\nRed Team:")
    for r_idx, role in enumerate(roles):
        champ = red_team.get(role)
        idx = get_champion_by_name(champ, champ_to_idx) if champ else None
        if idx is not None:
            X_deep[5 + r_idx] = idx
            X_wide[r_idx * num_champs + idx] = -1.0
            print(f"  {role.upper():7s}: {champion_names[idx]}")
        else:
            print(f"  {role.upper():7s}: Warning: '{champ}' not found in model.")
 
    # Convert vectors to PyTorch Tensors
    w_tensor = torch.tensor(X_wide, dtype=torch.float32).unsqueeze(0).to(device)
    d_tensor = torch.tensor(X_deep, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        probability = model(w_tensor, d_tensor).item()

    print(f"\n--- Result (Wide & Deep Attention Network) ---")
    print(f"Win Probability (Blue): {probability:.2%}")
    print(f"Win Probability (Red):  {(1-probability):.2%}")
    print(f"Predicted Winner: {'BLUE' if probability > 0.5 else 'RED'}")

if __name__ == "__main__":
    # Sample Test Case
    blue = {
        'top': 'sona',
        'jungle': 'syndra',
        'mid': 'udyr',
        'bot': 'vi',
        'support': 'viego'
    }
    red = {
        'top': 'vayne',
        'jungle': 'voli',
        'mid': 'zilean',
        'bot': 'zoe',
        'support': 'zac'
    }
    predict_match(blue, red)

import http.server
import socketserver
import json
import os
import torch
import torch.nn as nn
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get('PORT', 8000))

# Recreate WideAndDeepDraftNN architecture dynamically
class WideAndDeepDraftNN(nn.Module):
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

# Load model weights and configs on startup
model_path = os.path.join(SCRIPT_DIR, 'FFN', 'model_nn.pth')
meta_path = os.path.join(SCRIPT_DIR, 'FFN', 'model_nn_metadata.json')

if os.path.exists(model_path) and os.path.exists(meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    champion_names = meta['champion_names']
    champ_to_idx = meta['champ_to_idx']
    embedding_dim = meta.get('embedding_dim', 16)
    num_heads = meta.get('num_heads', 1)
    num_champs = len(champion_names)
    
    model = WideAndDeepDraftNN(num_champs, embedding_dim, num_heads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("PyTorch model loaded successfully on startup.")
else:
    print("Warning: FFN model/metadata not found. Run tune_nn.py first.")
    model = None
    champion_names = []
    champ_to_idx = {}
    num_champs = 0
    device = torch.device('cpu')

def get_champion_by_name(name, champ_to_idx):
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

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        routes = {
            '/':              ('text/html',              'index.html'),
            '/index.html':    ('text/html',              'index.html'),
            '/static/style.css':  ('text/css',          'static/style.css'),
            '/static/script.js':  ('application/javascript', 'static/script.js'),
            '/champions.json':    ('application/json',   'champions.json'),
        }
        if self.path not in routes:
            self.send_response(404)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            return
        content_type, rel_path = routes[self.path]
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        with open(os.path.join(SCRIPT_DIR, rel_path), 'rb') as f:
            self.wfile.write(f.read())

    def do_POST(self):
        if not model:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Model not loaded"}).encode('utf-8'))
            return

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        if self.path == '/api/predict':
            blue_team = {k.lower(): v for k, v in data.get('blue_team', {}).items()}
            red_team = {k.lower(): v for k, v in data.get('red_team', {}).items()}
            
            X_wide = np.zeros(num_champs * 5, dtype=np.float32)
            X_deep = np.zeros(10, dtype=np.int64)
            X_deep.fill(num_champs)
            
            roles = ['top', 'jungle', 'mid', 'bot', 'support']
            blue_resolved = {}
            red_resolved = {}
            
            for r_idx, role in enumerate(roles):
                champ = blue_team.get(role, '')
                idx = get_champion_by_name(champ, champ_to_idx)
                if idx is not None:
                    X_deep[r_idx] = idx
                    X_wide[r_idx * num_champs + idx] = 1.0
                    blue_resolved[role] = champion_names[idx]
                else:
                    blue_resolved[role] = champ if champ else "Empty"
                    
            for r_idx, role in enumerate(roles):
                champ = red_team.get(role, '')
                idx = get_champion_by_name(champ, champ_to_idx)
                if idx is not None:
                    X_deep[5 + r_idx] = idx
                    X_wide[r_idx * num_champs + idx] = -1.0
                    red_resolved[role] = champion_names[idx]
                else:
                    red_resolved[role] = champ if champ else "Empty"
            
            w_tensor = torch.tensor(X_wide, dtype=torch.float32).unsqueeze(0).to(device)
            d_tensor = torch.tensor(X_deep, dtype=torch.long).unsqueeze(0).to(device)
            
            with torch.no_grad():
                prob = model(w_tensor, d_tensor).item()
                
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "probability": prob,
                "blue_roster": blue_resolved,
                "red_roster": red_resolved
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))

        elif self.path == '/api/recommend':
            blue_team = {k.lower(): v for k, v in data.get('blue_team', {}).items()}
            red_team = {k.lower(): v for k, v in data.get('red_team', {}).items()}
            user_side = data.get('user_side', 'blue').lower()
            user_role = data.get('user_role', 'mid').lower()
            candidates = data.get('candidates', [])

            roles = ['top', 'jungle', 'mid', 'bot', 'support']
            if user_role not in roles:
                self.send_response(400)
                self.end_headers()
                return

            target_role_idx = roles.index(user_role)

            # Initialize base features (ignoring the slot we are recommending for)
            base_X_wide = np.zeros(num_champs * 5, dtype=np.float32)
            base_X_deep = np.zeros(10, dtype=np.int64)
            base_X_deep.fill(num_champs)

            for r_idx, role in enumerate(roles):
                if user_side == 'blue' and role == user_role:
                    continue
                champ = blue_team.get(role, '')
                idx = get_champion_by_name(champ, champ_to_idx)
                if idx is not None:
                    base_X_deep[r_idx] = idx
                    base_X_wide[r_idx * num_champs + idx] = 1.0

            for r_idx, role in enumerate(roles):
                if user_side == 'red' and role == user_role:
                    continue
                champ = red_team.get(role, '')
                idx = get_champion_by_name(champ, champ_to_idx)
                if idx is not None:
                    base_X_deep[5 + r_idx] = idx
                    base_X_wide[r_idx * num_champs + idx] = -1.0

            # Resolve valid candidate IDs
            valid_candidates = []
            valid_indices = []
            for name in candidates:
                idx = get_champion_by_name(name, champ_to_idx)
                if idx is not None:
                    valid_candidates.append(name)
                    valid_indices.append(idx)

            if not valid_candidates:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"recommendations": []}).encode('utf-8'))
                return

            # Build batch array for unified GPU inference
            N = len(valid_candidates)
            batch_X_wide = np.tile(base_X_wide, (N, 1))
            batch_X_deep = np.tile(base_X_deep, (N, 1))

            for i, idx in enumerate(valid_indices):
                if user_side == 'blue':
                    batch_X_deep[i, target_role_idx] = idx
                    batch_X_wide[i, target_role_idx * num_champs + idx] = 1.0
                else:
                    batch_X_deep[i, 5 + target_role_idx] = idx
                    batch_X_wide[i, target_role_idx * num_champs + idx] = -1.0

            w_tensor = torch.tensor(batch_X_wide, dtype=torch.float32).to(device)
            d_tensor = torch.tensor(batch_X_deep, dtype=torch.long).to(device)

            with torch.no_grad():
                probabilities = model(w_tensor, d_tensor).flatten().cpu().tolist()

            # Format user relative win rate responses
            results = []
            for name, prob in zip(valid_candidates, probabilities):
                user_win_rate = prob if user_side == 'blue' else (1.0 - prob)
                results.append({
                    "name": name,
                    "win_rate": user_win_rate
                })

            # Sort best win rate recommendation first
            results.sort(key=lambda x: x['win_rate'], reverse=True)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"recommendations": results}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def main():
    handler = RequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"\n=======================================================")
        print(f"  League Draft Predictor Client Server Active!")
        print(f"  Local Address: http://localhost:{PORT}")
        print(f"=======================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.server_close()

if __name__ == "__main__":
    main()

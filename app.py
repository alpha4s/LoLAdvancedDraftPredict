import http.server, socketserver, json, os, torch, numpy as np
from model import WideAndDeepDraftNN
from feature_engineering import load_feature_matrices, extract_draft_features

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get('PORT', 8000))

model_path = os.path.join(SCRIPT_DIR, 'FFN', 'model_nn.pth')
meta_path = os.path.join(SCRIPT_DIR, 'FFN', 'model_nn_metadata.json')

feature_matrices = load_feature_matrices()

if os.path.exists(model_path) and os.path.exists(meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    champion_names = meta['champion_names']
    champ_to_idx = meta['champ_to_idx']
    num_champs = len(champion_names)
    role_affinity = meta.get('role_affinity')
    scaler_dict = meta.get('feature_scaler')
    
    if scaler_dict:
        feature_mean = np.array(scaler_dict['mean'], dtype=np.float32)
        feature_std = np.array(scaler_dict['std'], dtype=np.float32)
    else:
        feature_mean, feature_std = np.zeros((1, 32), dtype=np.float32), np.ones((1, 32), dtype=np.float32)

    model = WideAndDeepDraftNN(num_champs, meta.get('embedding_dim', 16), meta.get('num_heads', 2), num_extra_features=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    print("PyTorch Wide & Deep model loaded successfully with 32 per-slot and team engineered features.")
else:
    model, champion_names, champ_to_idx, num_champs = None, [], {}, 0
    feature_mean, feature_std = np.zeros((1, 32), dtype=np.float32), np.ones((1, 32), dtype=np.float32)
    device = torch.device('cpu')

def get_champion_by_name(name, champ_to_idx):
    if not name: return None
    norm = "".join(c for c in name.lower() if c.isalnum())
    if not norm: return None
    norm_map = {"".join(c for c in k.lower() if c.isalnum()): v for k, v in champ_to_idx.items()}
    if norm in norm_map: return norm_map[norm]
    if len(norm) >= 4:
        for k, v in norm_map.items():
            if norm in k: return v
    return None

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass

    def send_resp(self, status, content_type, body):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        if body is not None:
            self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        routes = {
            '/': ('text/html', 'index.html'),
            '/index.html': ('text/html', 'index.html'),
            '/static/style.css': ('text/css', 'static/style.css'),
            '/static/script.js': ('application/javascript', 'static/script.js'),
            '/champions.json': ('application/json', 'champions.json'),
        }
        clean_path = self.path.split('?')[0]
        if clean_path not in routes:
            return self.send_resp(404, 'text/plain', b'Not Found')
        ctype, rpath = routes[clean_path]
        with open(os.path.join(SCRIPT_DIR, rpath), 'rb') as f:
            self.send_resp(200, ctype, f.read())

    def do_POST(self):
        if not model:
            return self.send_resp(500, 'application/json', {"error": "Model not loaded"})

        content_length = int(self.headers['Content-Length'])
        data = json.loads(self.rfile.read(content_length).decode('utf-8'))
        roles = ['top', 'jungle', 'mid', 'bot', 'support']

        if self.path == '/api/predict':
            teams = {'blue': data.get('blue_team', {}), 'red': data.get('red_team', {})}
            X_deep = np.full(10, num_champs, dtype=np.int64)
            rosters = {'blue_roster': {}, 'red_roster': {}}

            blue_names = [teams['blue'].get(r, '') for r in roles]
            red_names = [teams['red'].get(r, '') for r in roles]

            for t_idx, side in enumerate(['blue', 'red']):
                team = {k.lower(): v for k, v in teams[side].items()}
                for r_idx, role in enumerate(roles):
                    champ = team.get(role, '')
                    idx = get_champion_by_name(champ, champ_to_idx)
                    if idx is not None:
                        X_deep[t_idx * 5 + r_idx] = idx
                        rosters[f'{side}_roster'][role] = champion_names[idx]
                    else:
                        rosters[f'{side}_roster'][role] = champ if champ else "Empty"

            raw_feats = extract_draft_features(blue_names, red_names, feature_matrices)
            norm_feats = (raw_feats.reshape(1, -1) - feature_mean) / feature_std

            d_tensor = torch.tensor(X_deep, dtype=torch.long).unsqueeze(0).to(device)
            f_tensor = torch.tensor(norm_feats, dtype=torch.float32).to(device)

            with torch.no_grad():
                prob = model(d_tensor, f_tensor).item()

            return self.send_resp(200, 'application/json', {"probability": prob, **rosters})

        elif self.path == '/api/recommend':
            user_side = data.get('user_side', 'blue').lower()
            user_role = data.get('user_role', 'mid').lower()
            if user_role not in roles:
                return self.send_resp(400, 'text/plain', b'Invalid role')

            target_role_idx = roles.index(user_role)
            candidates = data.get('candidates', [])

            base_X_deep = np.full(10, num_champs, dtype=np.int64)
            teams = {'blue': data.get('blue_team', {}), 'red': data.get('red_team', {})}

            base_blue_names = [teams['blue'].get(r, '') for r in roles]
            base_red_names = [teams['red'].get(r, '') for r in roles]

            for t_idx, side in enumerate(['blue', 'red']):
                team = {k.lower(): v for k, v in teams[side].items()}
                for r_idx, role in enumerate(roles):
                    if side == user_side and role == user_role:
                        continue
                    champ = team.get(role, '')
                    idx = get_champion_by_name(champ, champ_to_idx)
                    if idx is not None:
                        base_X_deep[t_idx * 5 + r_idx] = idx

            valid_candidates, valid_indices = [], []
            for name in candidates:
                idx = get_champion_by_name(name, champ_to_idx)
                if idx is not None:
                    valid_candidates.append(name)
                    valid_indices.append(idx)

            if not valid_candidates:
                return self.send_resp(200, 'application/json', {"recommendations": []})

            N = len(valid_candidates)
            batch_X_deep = np.tile(base_X_deep, (N, 1))
            batch_raw_feats = []

            for i, (name, idx) in enumerate(zip(valid_candidates, valid_indices)):
                slot_offset = 0 if user_side == 'blue' else 5
                batch_X_deep[i, slot_offset + target_role_idx] = idx

                cand_blue = list(base_blue_names)
                cand_red = list(base_red_names)
                if user_side == 'blue':
                    cand_blue[target_role_idx] = name
                else:
                    cand_red[target_role_idx] = name

                batch_raw_feats.append(extract_draft_features(cand_blue, cand_red, feature_matrices))

            feats_mat = np.array(batch_raw_feats, dtype=np.float32)
            norm_feats = (feats_mat - feature_mean) / feature_std

            batch_d_tensor = torch.tensor(batch_X_deep, dtype=torch.long).to(device)
            batch_f_tensor = torch.tensor(norm_feats, dtype=torch.float32).to(device)

            with torch.no_grad():
                probs = model(batch_d_tensor, batch_f_tensor).squeeze(-1).cpu().numpy()

            recs = [{"name": cand, "win_rate": float(probs[i])} for i, cand in enumerate(valid_candidates)]
            recs.sort(key=lambda x: x['win_rate'], reverse=(user_side == 'blue'))

            return self.send_resp(200, 'application/json', {"recommendations": recs[:10]})

        else:
            return self.send_resp(404, 'text/plain', b'Not Found')

def main():
    with socketserver.TCPServer(("", PORT), RequestHandler) as httpd:
        print(f"League Draft Predictor Server running on http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.server_close()

if __name__ == "__main__":
    main()

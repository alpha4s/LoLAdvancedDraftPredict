import os, json, sqlite3, torch, argparse, pandas as pd, numpy as np, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

def load_data_from_db():
    db_path = os.path.join(ROOT_DIR, 'league_data.db')
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM matches", conn)
    conn.close()
    return df

def get_champion_metadata():
    json_path = os.path.join(ROOT_DIR, 'champions.json')
    with open(json_path, 'r') as f:
        champions_data = json.load(f)
    names = sorted(list(champions_data.values())) if isinstance(champions_data, dict) else sorted(list(champions_data))
    name_to_idx = {name: i for i, name in enumerate(names)}
    return names, name_to_idx

def vectorize_data(df, name_to_idx):
    num_matches = len(df)
    num_champs = len(name_to_idx)
    X_deep = np.zeros((num_matches, 10), dtype=np.int64)
    X_deep.fill(num_champs)
    X_wide = np.zeros((num_matches, num_champs * 5), dtype=np.float32)
    y = np.zeros(num_matches)
    roles = ['top', 'jungle', 'mid', 'bot', 'support']
    for i, row in df.iterrows():
        for r_idx, role in enumerate(roles):
            blue_champ = row[f'blue_{role}']
            if blue_champ in name_to_idx:
                idx = name_to_idx[blue_champ]
                X_deep[i, r_idx] = idx
                X_wide[i, r_idx * num_champs + idx] = 1.0
            red_champ = row[f'red_{role}']
            if red_champ in name_to_idx:
                idx = name_to_idx[red_champ]
                X_deep[i, 5 + r_idx] = idx
                X_wide[i, r_idx * num_champs + idx] = -1.0
        y[i] = 1 if row['winning_team'] == 'BLUE_WIN' else 0
    return X_wide, X_deep, y

class WideAndDeepDraftNN(nn.Module):
    def __init__(self, num_champs, embedding_dim=16, num_heads=1):
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

def train_and_evaluate(X_wide_train, X_deep_train, y_train, X_wide_val, X_deep_val, y_val, num_champs, embedding_dim, num_heads, alpha, device, epochs=100, patience=15):
    train_dataset = TensorDataset(
        torch.tensor(X_wide_train, dtype=torch.float32),
        torch.tensor(X_deep_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    model = WideAndDeepDraftNN(num_champs, embedding_dim, num_heads).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=alpha)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for w_in, d_in, targets in train_loader:
            w_in, d_in, targets = w_in.to(device), d_in.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(w_in, d_in)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            w_val = torch.tensor(X_wide_val, dtype=torch.float32).to(device)
            d_val = torch.tensor(X_deep_val, dtype=torch.long).to(device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
            val_loss = criterion(model(w_val, d_val), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        w_val = torch.tensor(X_wide_val, dtype=torch.float32).to(device)
        d_val = torch.tensor(X_deep_val, dtype=torch.long).to(device)
        preds = (model(w_val, d_val).cpu().numpy() > 0.5).astype(int)
        score = accuracy_score(y_val, preds)

    return model, score

def save_files(model, champion_names, name_to_idx, score, embed_dim, num_heads, alpha):
    model_path = os.path.join(SCRIPT_DIR, 'model_nn.pth')
    torch.save(model.state_dict(), model_path)
    metadata = {
        'champion_names': champion_names,
        'champ_to_idx': name_to_idx,
        'accuracy': score,
        'model_type': 'PyTorch_WideAndDeepAttentionDraftNN',
        'architecture': f"Wide(Linear) + Deep(Embedding({embed_dim}) -> SelfAttention({num_heads}heads) -> Linear(32) -> 1)",
        'embedding_dim': embed_dim,
        'num_heads': num_heads,
        'best_hyperparameters': {
            'embedding_dim': embed_dim,
            'num_heads': num_heads,
            'alpha': alpha
        }
    }
    meta_path = os.path.join(SCRIPT_DIR, 'model_nn_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"\nSaved model_nn.pth and updated model_nn_metadata.json (Accuracy: {score:.2%})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    df = load_data_from_db()
    champ_names, name_to_idx = get_champion_metadata()
    X_wide, X_deep, y = vectorize_data(df, name_to_idx)

    if len(y) >= 70000:
        X_wide_train, X_deep_train, y_train = X_wide[:50000], X_deep[:50000], y[:50000]
        X_wide_val, X_deep_val, y_val = X_wide[50000:70000], X_deep[50000:70000], y[50000:70000]
    else:
        X_wide_train, X_wide_val, X_deep_train, X_deep_val, y_train, y_val = train_test_split(
            X_wide, X_deep, y, test_size=0.2, random_state=42
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    num_champs = len(champ_names)

    if args.tune:
        print("Tuning hyperparameters...")
        embedding_configs = [(8, 1), (8, 2), (16, 1), (16, 2)]
        alphas = [1e-6, 1e-5, 1e-4]
        best_acc, best_config, best_model = 0.0, {}, None

        for embed_dim, num_heads in embedding_configs:
            for alpha in alphas:
                print(f"Testing Config: EmbedDim={embed_dim}, Heads={num_heads}, Alpha={alpha}")
                model, score = train_and_evaluate(
                    X_wide_train, X_deep_train, y_train,
                    X_wide_val, X_deep_val, y_val,
                    num_champs, embed_dim, num_heads, alpha, device
                )
                print(f" -> Validation Accuracy: {score:.2%}")
                if score > best_acc:
                    best_acc = score
                    best_config = {'embedding_dim': embed_dim, 'num_heads': num_heads, 'alpha': alpha}
                    best_model = model

        print(f"\nBest Config found: {best_config} | Accuracy: {best_acc:.2%}")
        save_files(best_model, champ_names, name_to_idx, best_acc, best_config['embedding_dim'], best_config['num_heads'], best_config['alpha'])
    else:
        print("Training model with standard hyperparameters...")
        model, score = train_and_evaluate(
            X_wide_train, X_deep_train, y_train,
            X_wide_val, X_deep_val, y_val,
            num_champs, 16, 1, 1e-5, device, epochs=500, patience=20
        )
        save_files(model, champ_names, name_to_idx, score, 16, 1, 1e-5)

if __name__ == "__main__":
    main()

import os, json, sqlite3, torch, argparse, pandas as pd, numpy as np, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)

from model import WideAndDeepDraftNN

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

def build_adj_matrix(df, name_to_idx):
    """Builds pairwise synergy & counter champion adjacency matrix from historical matches."""
    num_champs = len(name_to_idx)
    adj = np.zeros((num_champs + 1, num_champs + 1), dtype=np.float32)
    roles = ['top', 'jungle', 'mid', 'bot', 'support']

    for _, row in df.iterrows():
        b_indices = [name_to_idx[row[f'blue_{r}']] for r in roles if row[f'blue_{r}'] in name_to_idx]
        r_indices = [name_to_idx[row[f'red_{r}']] for r in roles if row[f'red_{r}'] in name_to_idx]

        # Blue Team Synergy
        for i in range(len(b_indices)):
            for j in range(i + 1, len(b_indices)):
                adj[b_indices[i], b_indices[j]] += 1.0
                adj[b_indices[j], b_indices[i]] += 1.0

        # Red Team Synergy
        for i in range(len(r_indices)):
            for j in range(i + 1, len(r_indices)):
                adj[r_indices[i], r_indices[j]] += 1.0
                adj[r_indices[j], r_indices[i]] += 1.0

        # Blue vs Red Counters
        for b_idx in b_indices:
            for r_idx in r_indices:
                adj[b_idx, r_idx] += 0.5
                adj[r_idx, b_idx] += 0.5

    # Self-loops & Row normalization
    np.fill_diagonal(adj, 1.0)
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj_norm = adj / row_sums
    return adj_norm

def vectorize_data(df, name_to_idx, augment_snake_draft=True):
    """
    Vectorizes matches into X_wide, X_deep, and y tensors.
    Augments dataset with Snake-Draft partial states so the model accurately evaluates mid-draft turns.
    """
    num_champs = len(name_to_idx)
    roles = ['top', 'jungle', 'mid', 'bot', 'support']
    
    # Official Snake Draft Pick Stages: (Blue Picks, Red Picks)
    # Turn 1: B1 | Turn 2: B1, R1, R2 | Turn 3: B1-B3, R1-R2 | Turn 4: B1-B3, R1-R4 | Turn 5: B1-B5, R1-R4 | Turn 6: Full 5v5
    snake_stages = [
        ([0], []),
        ([0], [0, 1]),
        ([0, 1, 2], [0, 1]),
        ([0, 1, 2], [0, 1, 2, 3]),
        ([0, 1, 2, 3, 4], [0, 1, 2, 3]),
        ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) # Full match
    ]

    X_wide_list, X_deep_list, y_list = [], [], []

    for _, row in df.iterrows():
        b_indices = [name_to_idx.get(row[f'blue_{r}'], num_champs) for r in roles]
        r_indices = [name_to_idx.get(row[f'red_{r}'], num_champs) for r in roles]
        win = 1.0 if row['winning_team'] == 'BLUE_WIN' else 0.0

        stages_to_use = snake_stages if augment_snake_draft else [snake_stages[-1]]
        for b_active_roles, r_active_roles in stages_to_use:
            x_d = np.full(10, num_champs, dtype=np.int64)
            x_w = np.zeros(num_champs * 5, dtype=np.float32)

            for r_idx in b_active_roles:
                c_idx = b_indices[r_idx]
                if c_idx < num_champs:
                    x_d[r_idx] = c_idx
                    x_w[r_idx * num_champs + c_idx] = 1.0

            for r_idx in r_active_roles:
                c_idx = r_indices[r_idx]
                if c_idx < num_champs:
                    x_d[5 + r_idx] = c_idx
                    x_w[r_idx * num_champs + c_idx] = -1.0

            X_deep_list.append(x_d)
            X_wide_list.append(x_w)
            y_list.append(win)

    return np.array(X_wide_list, dtype=np.float32), np.array(X_deep_list, dtype=np.int64), np.array(y_list, dtype=np.float32)

def train_and_evaluate(X_wide_train, X_deep_train, y_train, X_wide_val, X_deep_val, y_val, num_champs, embedding_dim, num_heads, alpha, device, adj_matrix, epochs=100, patience=15):
    train_dataset = TensorDataset(
        torch.tensor(X_wide_train, dtype=torch.float32),
        torch.tensor(X_deep_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    model = WideAndDeepDraftNN(num_champs, embedding_dim, num_heads, adj_matrix=adj_tensor).to(device)
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

def save_files(model, champion_names, name_to_idx, score, embed_dim, num_heads, alpha, adj_matrix):
    model_path = os.path.join(SCRIPT_DIR, 'model_nn.pth')
    torch.save(model.state_dict(), model_path)
    metadata = {
        'champion_names': champion_names,
        'champ_to_idx': name_to_idx,
        'accuracy': score,
        'model_type': 'PyTorch_GNN_WideAndDeepAttentionDraftNN',
        'architecture': f"GCN(PairwiseAdj) + Wide(Linear) + Deep(Embedding({embed_dim}) -> SelfAttention({num_heads}heads) -> Linear(32) -> 1)",
        'embedding_dim': embed_dim,
        'num_heads': num_heads,
        'adj_matrix': adj_matrix.tolist(),
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
    
    print("Building champion synergy & counter GNN adjacency matrix...")
    adj_matrix = build_adj_matrix(df, name_to_idx)
    
    print("Vectorizing draft dataset with Snake-Draft partial state augmentation...")
    X_wide, X_deep, y = vectorize_data(df, name_to_idx, augment_snake_draft=True)

    if len(y) >= 70000:
        X_wide_train, X_deep_train, y_train = X_wide[:50000], X_deep[:50000], y[:50000]
        X_wide_val, X_deep_val, y_val = X_wide[50000:70000], X_deep[50000:70000], y[50000:70000]
    else:
        X_wide_train, X_wide_val, X_deep_train, X_deep_val, y_train, y_val = train_test_split(
            X_wide, X_deep, y, test_size=0.2, random_state=42
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Augmented Training Samples: {len(y_train)}")
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
                    num_champs, embed_dim, num_heads, alpha, device, adj_matrix
                )
                print(f" -> Validation Accuracy: {score:.2%}")
                if score > best_acc:
                    best_acc = score
                    best_config = {'embedding_dim': embed_dim, 'num_heads': num_heads, 'alpha': alpha}
                    best_model = model

        print(f"\nBest Config found: {best_config} | Accuracy: {best_acc:.2%}")
        save_files(best_model, champ_names, name_to_idx, best_acc, best_config['embedding_dim'], best_config['num_heads'], best_config['alpha'], adj_matrix)
    else:
        print("Training model with standard hyperparameters...")
        model, score = train_and_evaluate(
            X_wide_train, X_deep_train, y_train,
            X_wide_val, X_deep_val, y_val,
            num_champs, 16, 1, 1e-5, device, adj_matrix, epochs=100, patience=15
        )
        save_files(model, champ_names, name_to_idx, score, 16, 1, 1e-5, adj_matrix)

if __name__ == "__main__":
    main()

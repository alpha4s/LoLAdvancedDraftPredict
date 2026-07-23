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

def build_adj_matrix(df_train, name_to_idx):
    """
    Builds pairwise synergy & counter champion adjacency matrix using TRAINING matches only to prevent leakage.
    Returns un-normalized raw count adjacency matrix (model.py handles degree normalization).
    """
    num_champs = len(name_to_idx)
    adj = np.zeros((num_champs + 1, num_champs + 1), dtype=np.float32)
    roles = ['top', 'jungle', 'mid', 'bot', 'support']

    for _, row in df_train.iterrows():
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

    # Self-loops on active champions
    np.fill_diagonal(adj, 1.0)
    return adj

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
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=alpha)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for w_in, d_in, targets in train_loader:
            w_in, d_in, targets = w_in.to(device), d_in.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(w_in, d_in)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)

        train_loss /= len(train_dataset)

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
            improved_flag = " (Best)"
        else:
            patience_counter += 1
            improved_flag = ""

        print(f"  Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{improved_flag} | Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"  Early stopping triggered at Epoch {epoch}.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    stage_scores = {}
    with torch.no_grad():
        w_val = torch.tensor(X_wide_val, dtype=torch.float32).to(device)
        d_val = torch.tensor(X_deep_val, dtype=torch.long).to(device)
        preds = (model(w_val, d_val).cpu().numpy() > 0.5).astype(int)
        score = accuracy_score(y_val, preds)

        stage_names = [
            "Turn 1 (1 Pick)",
            "Turn 2 (3 Picks)",
            "Turn 3 (5 Picks)",
            "Turn 4 (7 Picks)",
            "Turn 5 (9 Picks)",
            "Turn 6 (Full 5v5 Draft)"
        ]
        if len(y_val) % 6 == 0:
            for s_idx, s_name in enumerate(stage_names):
                idx = np.arange(s_idx, len(y_val), 6)
                s_acc = accuracy_score(y_val[idx], preds[idx])
                stage_scores[s_name] = s_acc

    return model, score, stage_scores

def save_files(model, champion_names, name_to_idx, score, stage_scores, embed_dim, num_heads, alpha, adj_matrix):
    model_path = os.path.join(SCRIPT_DIR, 'model_nn.pth')
    torch.save(model.state_dict(), model_path)
    metadata = {
        'champion_names': champion_names,
        'champ_to_idx': name_to_idx,
        'accuracy': score,
        'stage_accuracies': stage_scores,
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

    print("\n=================== HELD-OUT ACCURACY BREAKDOWN ===================")
    print(f"Overall Blended Accuracy (Turns 1-6): {score:.2%}")
    if stage_scores:
        print("Accuracy by Draft Turn:")
        for name, s_acc in stage_scores.items():
            print(f"  - {name:25s}: {s_acc:.2%}")
    print("===================================================================")
    print(f"Saved model_nn.pth and updated model_nn_metadata.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--embed_dim", type=int, default=8, help="Embedding dimension (default: 8)")
    parser.add_argument("--alpha", type=float, default=1e-3, help="L2 weight decay regularization (default: 1e-3)")
    args = parser.parse_args()

    df = load_data_from_db()
    champ_names, name_to_idx = get_champion_metadata()
    
    # 1. Split df FIRST at the match level to prevent data leakage between train and val
    print("Splitting matches into clean train and held-out validation sets...")
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

    # 2. Build GNN Adjacency Matrix using TRAINING matches ONLY!
    print("Building champion synergy & counter GNN adjacency matrix from training set only...")
    adj_matrix = build_adj_matrix(df_train, name_to_idx)
    
    # 3. Vectorize df_train and df_val SEPARATELY
    print("Vectorizing draft datasets with Snake-Draft partial state augmentation...")
    X_wide_train, X_deep_train, y_train = vectorize_data(df_train, name_to_idx, augment_snake_draft=True)
    X_wide_val, X_deep_val, y_val = vectorize_data(df_val, name_to_idx, augment_snake_draft=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Train Augmented Samples: {len(y_train)} | Val Augmented Samples: {len(y_val)}")
    print(f"Blue Win Rate - Train: {y_train.mean():.2%} | Val: {y_val.mean():.2%}")
    num_champs = len(champ_names)

    if args.tune:
        print("Tuning hyperparameters across strong L2 regularization & embedding dimensions...")
        embedding_configs = [(8, 1), (8, 2), (16, 1), (16, 2)]
        alphas = [1e-4, 1e-3, 1e-2]
        best_acc, best_config, best_model, best_stages = 0.0, {}, None, {}

        for embed_dim, num_heads in embedding_configs:
            for alpha in alphas:
                print(f"\n--- Testing Config: EmbedDim={embed_dim}, Heads={num_heads}, Alpha={alpha} ---")
                model, score, stage_scores = train_and_evaluate(
                    X_wide_train, X_deep_train, y_train,
                    X_wide_val, X_deep_val, y_val,
                    num_champs, embed_dim, num_heads, alpha, device, adj_matrix, epochs=100, patience=15
                )
                print(f" -> Overall Validation Accuracy: {score:.2%}")
                if score > best_acc:
                    best_acc = score
                    best_config = {'embedding_dim': embed_dim, 'num_heads': num_heads, 'alpha': alpha}
                    best_model = model
                    best_stages = stage_scores

        print(f"\nBest Config found: {best_config} | Held-Out Accuracy: {best_acc:.2%}")
        save_files(best_model, champ_names, name_to_idx, best_acc, best_stages, best_config['embedding_dim'], best_config['num_heads'], best_config['alpha'], adj_matrix)
    else:
        print(f"Training model with EmbedDim={args.embed_dim}, Alpha={args.alpha}...")
        model, score, stage_scores = train_and_evaluate(
            X_wide_train, X_deep_train, y_train,
            X_wide_val, X_deep_val, y_val,
            num_champs, args.embed_dim, 1, args.alpha, device, adj_matrix, epochs=100, patience=15
        )
        save_files(model, champ_names, name_to_idx, score, stage_scores, args.embed_dim, 1, args.alpha, adj_matrix)

if __name__ == "__main__":
    main()

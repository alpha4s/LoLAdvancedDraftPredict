import os, json, sqlite3, torch, pandas as pd, numpy as np, torch.nn as nn, torch.optim as optim; from torch.utils.data import DataLoader, TensorDataset; from sklearn.model_selection import train_test_split; from sklearn.metrics import accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

def load_data_from_db():
    """Load historical match records from sqlite database."""
    db_path = os.path.join(ROOT_DIR, 'league_data.db')
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM matches", conn)
    conn.close()
    return df

def get_champion_metadata():
    """Load champion list and return sorted names list and name-to-index mapping."""
    json_path = os.path.join(ROOT_DIR, 'champions.json')
    with open(json_path, 'r') as f:
        champions_data = json.load(f)
    names = sorted(list(champions_data.values())) if isinstance(champions_data, dict) else sorted(list(champions_data))
    name_to_idx = {name: i for i, name in enumerate(names)}
    return names, name_to_idx

def vectorize_data(df, name_to_idx):
    """
    Format raw drafts into:
      1. X_deep: (N, 10) integer array representing champion IDs.
      2. X_wide: (N, 870) float array representing role-specific picks (+1 Blue, -1 Red).
    """
    num_matches = len(df)
    num_champs = len(name_to_idx)
    
    X_deep = np.zeros((num_matches, 10), dtype=np.int64)
    padding_idx = num_champs
    X_deep.fill(padding_idx)
    
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

def train_and_evaluate(X_wide_train, X_deep_train, y_train, X_wide_test, X_deep_test, y_test, num_champs, embedding_dim, num_heads, alpha, device):
    """Train a single model configuration on the training set and evaluate its validation accuracy."""
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
    patience = 15
    patience_counter = 0
    
    for epoch in range(120):
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
            w_test = torch.tensor(X_wide_test, dtype=torch.float32).to(device)
            d_test = torch.tensor(X_deep_test, dtype=torch.long).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
            val_outputs = model(w_test, d_test)
            val_loss = criterion(val_outputs, y_test_tensor).item()
            
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
        w_test = torch.tensor(X_wide_test, dtype=torch.float32).to(device)
        d_test = torch.tensor(X_deep_test, dtype=torch.long).to(device)
        preds = (model(w_test, d_test).cpu().numpy() > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        
    return model, acc

def main():
    print("Loading data...")
    df = load_data_from_db()
    champion_names, name_to_idx = get_champion_metadata()
    X_wide, X_deep, y = vectorize_data(df, name_to_idx)
    
    print(f"Loaded {len(y)} matches.")
    if len(y) < 70000:
        print("Error: You need at least 70,000 matches in league_data.db to perform the 50k/20k tune split.")
        return
        
    # Split exactly 50,000 matches for training and 20,000 matches for testing
    X_wide_train = X_wide[:50000]
    X_deep_train = X_deep[:50000]
    y_train = y[:50000]
    
    X_wide_test = X_wide[50000:70000]
    X_deep_test = X_deep[50000:70000]
    y_test = y[50000:70000]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for tuning: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Define hyperparameter configs to sweep sequentially
    embedding_configs = [(8, 1), (8, 2), (16, 1), (16, 2)]
    alphas = [1e-6, 1e-5, 1e-4]
    
    print("\nStarting PyTorch Wide & Deep Attention Grid Search Tuning...")
    print("Running sequential iterations on the GPU (highly accelerated)...")
    
    best_accuracy = 0.0
    best_config = {}
    best_model = None
    
    num_champs = len(champion_names)
    iteration = 1
    total_iterations = len(embedding_configs) * len(alphas)
    results = []
    
    for embed_dim, num_heads in embedding_configs:
        for alpha in alphas:
            print(f"\n[{iteration}/{total_iterations}] Testing config: EmbedDim={embed_dim}, Heads={num_heads}, Alpha={alpha}")
            
            model, val_acc = train_and_evaluate(
                X_wide_train, X_deep_train, y_train, 
                X_wide_test, X_deep_test, y_test,
                num_champs, embed_dim, num_heads, alpha, device
            )
            
            print(f"-> Accuracy: {val_acc:.2%}")
            results.append((embed_dim, num_heads, alpha, val_acc))
            
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_config = {
                    'embedding_dim': embed_dim,
                    'num_heads': num_heads,
                    'alpha': alpha
                }
                best_model = model
                
            iteration += 1
                
    print("\n=== Attention Tuning Grid Results Summary ===")
    print(f"{'Embed Dimension':15s} | {'Heads':6s} | {'Weight Decay (Alpha)':20s} | {'Accuracy':10s}")
    print("-" * 62)
    for embed_dim, num_heads, alp, acc in results:
        print(f"{embed_dim:<15d} | {num_heads:<6d} | {alp:<20f} | {acc:.2%}")
        
    print("\n=== Best Model Found ===")
    print(f"Best Configuration: {best_config}")
    print(f"Best Accuracy: {best_accuracy:.2%}")
    
    # Save the winning weights
    model_path = os.path.join(SCRIPT_DIR, 'model_nn.pth')
    torch.save(best_model.state_dict(), model_path)
    
    # Write metadata
    arch_str = f"Wide(Linear) + Deep(Embedding({best_config['embedding_dim']}) -> SelfAttention({best_config['num_heads']}heads) -> Linear(32) -> 1)"
    metadata = {
        'champion_names': champion_names,
        'champ_to_idx': name_to_idx,
        'accuracy': best_accuracy,
        'model_type': 'PyTorch_WideAndDeepAttentionDraftNN',
        'architecture': arch_str,
        'embedding_dim': best_config['embedding_dim'],
        'num_heads': best_config['num_heads'],
        'best_hyperparameters': best_config
    }
    meta_path = os.path.join(SCRIPT_DIR, 'model_nn_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("\nSuccess! Best PyTorch Wide & Deep Attention model saved to model_nn.pth and updated model_nn_metadata.json.")

if __name__ == "__main__":
    main()

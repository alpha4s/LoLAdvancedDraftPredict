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
    
    # Sequence mapping for deep embeddings
    X_deep = np.zeros((num_matches, 10), dtype=np.int64)
    padding_idx = num_champs
    X_deep.fill(padding_idx)
    
    # Flat sparse array for linear wide component
    X_wide = np.zeros((num_matches, num_champs * 5), dtype=np.float32)
    y = np.zeros(num_matches)
    
    roles = ['top', 'jungle', 'mid', 'bot', 'support']
    for i, row in df.iterrows():
        for r_idx, role in enumerate(roles):
            # Blue picks encoded as positive
            blue_champ = row[f'blue_{role}']
            if blue_champ in name_to_idx:
                idx = name_to_idx[blue_champ]
                X_deep[i, r_idx] = idx
                X_wide[i, r_idx * num_champs + idx] = 1.0
                
            # Red picks encoded as negative
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
    def __init__(self, num_champs, embedding_dim=16, num_heads=1):
        super(WideAndDeepDraftNN, self).__init__()
        self.num_champs = num_champs
        self.embedding_dim = embedding_dim
        
        # 1. Wide Linear model
        self.wide_linear = nn.Linear(num_champs * 5, 1, bias=False)
        
        # 2. Deep Embedding layers
        self.champ_embeddings = nn.Embedding(num_champs + 1, embedding_dim, padding_idx=num_champs)
        self.role_embeddings = nn.Embedding(5, embedding_dim)
        
        # 3. Deep Self-Attention layers
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True, dropout=0.2)
        self.ln2 = nn.LayerNorm(2 * embedding_dim)
        
        # 4. Dense prediction layers
        self.fc = nn.Sequential(
            nn.Linear(2 * embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_wide, x_deep):
        # Compute linear logit contribution
        wide_out = self.wide_linear(x_wide)
        
        # Compute deep attention logit contribution
        batch_size = x_deep.size(0)
        device = x_deep.device
        
        # Fetch embeddings and add positional role vectors
        champs = self.champ_embeddings(x_deep)
        role_idx = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long, device=device)
        role_idx = role_idx.unsqueeze(0).expand(batch_size, -1)
        roles = self.role_embeddings(role_idx)
        seq = champs + roles
        
        # Run self-attention block with residual skip connection
        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.ln1(seq + attn_out)
        
        # Average pooling across roles for both teams
        blue_rep = seq[:, :5, :].mean(dim=1)
        red_rep = seq[:, 5:, :].mean(dim=1)
        combined = torch.cat([blue_rep, red_rep], dim=1)
        combined = self.ln2(combined)
        
        deep_out = self.fc(combined)
        
        # Merge wide/deep outputs and scale to probability
        return torch.sigmoid(wide_out + deep_out)

def train_neural_network(X_wide, X_deep, y, champion_names):
    """Split dataset, train Wide & Deep network with early stopping on GPU, and evaluate test accuracy."""
    if len(y) >= 70000:
        print("Using exactly 50,000 matches for training and 20,000 for testing.")
        X_wide_train = X_wide[:50000]
        X_deep_train = X_deep[:50000]
        y_train = y[:50000]
        
        X_wide_test = X_wide[50000:70000]
        X_deep_test = X_deep[50000:70000]
        y_test = y[50000:70000]
    else:
        print(f"Dataset has {len(y)} matches. Falling back to 80/20 split.")
        X_wide_train, X_wide_test, X_deep_train, X_deep_test, y_train, y_test = train_test_split(
            X_wide, X_deep, y, test_size=0.2, random_state=42
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Prepare datasets and loaders
    train_dataset = TensorDataset(
        torch.tensor(X_wide_train, dtype=torch.float32),
        torch.tensor(X_deep_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    num_champs = len(champion_names)
    model = WideAndDeepDraftNN(num_champs, embedding_dim=16, num_heads=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    best_val_loss = float('inf')
    best_model_state = None
    patience = 20
    patience_counter = 0

    print("Beginning PyTorch training epochs...")
    for epoch in range(500):
        model.train()
        train_loss = 0.0
        for w_in, d_in, targets in train_loader:
            w_in, d_in, targets = w_in.to(device), d_in.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(w_in, d_in)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * w_in.size(0)
        train_loss /= len(X_wide_train)

        # Run validation evaluation
        model.eval()
        with torch.no_grad():
            w_test_tensor = torch.tensor(X_wide_test, dtype=torch.float32).to(device)
            d_test_tensor = torch.tensor(X_deep_test, dtype=torch.long).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
            val_outputs = model(w_test_tensor, d_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor).item()

        # Track best state for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/500 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Validation Loss: {best_val_loss:.4f}")
            break

    # Restore best parameters
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluate final test accuracy
    model.eval()
    with torch.no_grad():
        w_test_tensor = torch.tensor(X_wide_test, dtype=torch.float32).to(device)
        d_test_tensor = torch.tensor(X_deep_test, dtype=torch.long).to(device)
        preds = (model(w_test_tensor, d_test_tensor).cpu().numpy() > 0.5).astype(int)
        score = accuracy_score(y_test, preds)

    print(f"\nWide & Deep Attention Network Accuracy: {score:.2%}")
    return model, score

def save_files(model, champion_names, name_to_idx, score):
    """Save PyTorch state dict and configurations metadata."""
    model_path = os.path.join(SCRIPT_DIR, 'model_nn.pth')
    torch.save(model.state_dict(), model_path)

    metadata = {
        'champion_names': champion_names,
        'champ_to_idx': name_to_idx,
        'accuracy': score,
        'model_type': 'PyTorch_WideAndDeepAttentionDraftNN',
        'architecture': 'Wide(Linear) + Deep(Embedding(16) -> SelfAttention(1head) -> Linear(32) -> 1)',
        'embedding_dim': 16,
        'num_heads': 1
    }
    meta_path = os.path.join(SCRIPT_DIR, 'model_nn_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("\nFiles saved: model_nn.pth and model_nn_metadata.json")

def main():
    df = load_data_from_db()
    champ_names, name_to_idx = get_champion_metadata()
    X_wide, X_deep, y = vectorize_data(df, name_to_idx)
    model, score = train_neural_network(X_wide, X_deep, y, champ_names)
    save_files(model, champ_names, name_to_idx, score)

if __name__ == "__main__":
    main()

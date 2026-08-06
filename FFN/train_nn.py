import copy
import os, sys, json, sqlite3, torch, pandas as pd, numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config import DB_PATH, CHAMPIONS_PATH, MODEL_WEIGHTS_PATH, MODEL_META_PATH, ONNX_MODEL_PATH, ROLES, EMBEDDING_DIM
from model import WideAndDeepDraftNN
from feature_engineering import build_feature_matrices

np.random.seed(42)
torch.manual_seed(42)

print("=== TRAINING WIDE-AND-DEEP DRAFT MODEL ===")

# 1. Load data directly from SQLite database
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM matches WHERE game_version LIKE ?", conn, params=("16%",))
conn.close()

# 2. champ mapping
with open(CHAMPIONS_PATH, 'r') as f:
    champs = sorted(json.load(f))
champ_to_idx = {c: i for i, c in enumerate(champs)}
num_champs = len(champs)

# 3. train / test split (80 / 20)
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
feature_matrices = build_feature_matrices(df_train)

# 4. feats to tables
zeros = lambda *shape: np.zeros(shape, dtype=np.float32)

ap_ratios = zeros(num_champs + 1, 1)
ap_variances = zeros(num_champs + 1, 1)
role_freqs = zeros(num_champs + 1, 5)

top_counters = zeros((num_champs + 1)**2, 1)
mid_counters = zeros((num_champs + 1)**2, 1)
supp_counters = zeros((num_champs + 1)**2, 1)

get_mat = lambda key: feature_matrices.get(key, {})

ap_ratio_dict = get_mat('champ_ap_ratios')
ap_variance_dict = get_mat('champ_ap_variances')
role_freq_dict = get_mat('role_freqs')
counter_stats_dict = get_mat('counter_matrix')

for champ, idx in champ_to_idx.items():
    ap_ratios[idx, 0] = float(ap_ratio_dict.get(champ, .5))
    ap_variances[idx, 0] = float(ap_variance_dict.get(champ, 0))
    for role_idx, role in enumerate(ROLES):
        role_freqs[idx, role_idx] = float(role_freq_dict.get(champ, {}).get(role, 0))

for champ_blue, idx_blue in champ_to_idx.items():
    for champ_red, idx_red in champ_to_idx.items():
        flat_idx = idx_blue * (num_champs + 1) + idx_red
        top_counters[flat_idx, 0] = float(counter_stats_dict.get(f"top:{champ_blue}_vs_{champ_red}", 0))
        mid_counters[flat_idx, 0] = float(counter_stats_dict.get(f"mid:{champ_blue}_vs_{champ_red}", 0))
        supp_counters[flat_idx, 0] = float(counter_stats_dict.get(f"support:{champ_blue}_vs_{champ_red}", 0))

to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32)

feature_tensors = {
    'ap_ratios': to_tensor(ap_ratios),
    'ap_variances': to_tensor(ap_variances),
    'role_freqs': to_tensor(role_freqs),
    'top_counters': to_tensor(top_counters),
    'mid_counters': to_tensor(mid_counters),
    'supp_counters': to_tensor(supp_counters)
}

from torch.utils.data import TensorDataset, DataLoader

def extract_champion_ids(df_subset, augment_partial=False):
    X_deep_list = []
    y_list = []
    winning_teams = df_subset['winning_team'].to_numpy()

    for i, (_, row) in enumerate(df_subset.iterrows()):
        b_champs = [row.get(f'blue_{r}', '') or '' for r in ROLES]
        r_champs = [row.get(f'red_{r}', '') or '' for r in ROLES]

        b_idxs = [champ_to_idx.get(c, num_champs) for c in b_champs]
        r_idxs = [champ_to_idx.get(c, num_champs) for c in r_champs]
        target = 1 if winning_teams[i] == 'BLUE_WIN' else 0

        # always include the complete draft
        X_deep_list.append(b_idxs + r_idxs)
        y_list.append(target)

        # sometimes include a partially filled draft for the live drafting UI
        if augment_partial and np.random.rand() < .5:
            n_b = np.random.randint(0, 6)
            n_r = np.random.randint(0, 6)

            b_partial = list(b_idxs)
            r_partial = list(r_idxs)

            # randomly remove picks from both teams
            if n_b < 5:
                mask_b = np.random.choice(5, 5 - n_b, replace=False)
                for idx in mask_b: b_partial[idx] = num_champs
            if n_r < 5:
                mask_r = np.random.choice(5, 5 - n_r, replace=False)
                for idx in mask_r: r_partial[idx] = num_champs

            X_deep_list.append(b_partial + r_partial)
            y_list.append(target)

    return np.array(X_deep_list, dtype=np.int64), np.array(y_list, dtype=np.float32)

print("Extracting champion IDs and partial-draft training examples...")
x_train_raw, y_train_raw = extract_champion_ids(df_train, augment_partial=True)
x_val_raw, y_val_raw = extract_champion_ids(df_val, augment_partial=False)
baseline_acc = max(float(y_val_raw.mean()), 1.0 - float(y_val_raw.mean()))
print(f"Majority-class baseline accuracy: {baseline_acc:.2%}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | Train Samples (Augmented): {len(y_train_raw)} | Val Samples: {len(y_val_raw)}")

device_feature_tensors = {key: tensor.to(device) for key, tensor in feature_tensors.items()}

# 5. initialize the model and training loop
epochs = 30
model = WideAndDeepDraftNN(num_champs=num_champs, features=device_feature_tensors, embed_dim=EMBEDDING_DIM).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

x_train_tensor = torch.tensor(x_train_raw, dtype=torch.long)
y_train_tensor = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.5)

x_val = torch.tensor(x_val_raw, dtype=torch.long, device=device)
y_val = torch.tensor(y_val_raw, dtype=torch.float32, device=device).unsqueeze(1)

best_val_loss = float('inf')
best_weights = None

print("\n--- Training wide-and-deep model (StepLR, gradient clipping) ---")
for epoch in range(1, epochs + 1):
    model.train()
    total_train_loss = 0.0
    for bx, by in train_loader:
        bx = bx.to(device)
        by = by.to(device)

        optimizer.zero_grad()
        preds = model(bx)
        loss = criterion(preds, by)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        total_train_loss += loss.item() * bx.size(0)

    scheduler.step()
    avg_train_loss = total_train_loss / len(train_dataset)

    model.eval()
    with torch.no_grad():
        val_preds = model(x_val)
        val_loss = criterion(val_preds, y_val).item()
        val_acc = accuracy_score(y_val_raw, (val_preds.cpu().numpy() > 0.5).astype(int))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = copy.deepcopy(model.state_dict())

    if epoch % 5 == 0 or epoch == epochs:
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

if best_weights is not None:
    model.load_state_dict(best_weights)

# evaluate final validation accuracy
model.eval()
with torch.no_grad():
    final_preds = (model(x_val).cpu().numpy() > 0.5).astype(int)
    final_acc = accuracy_score(y_val_raw, final_preds)

print("\n" + "="*50)
print(f"End-to-End Wide & Deep Full Draft (10/10) Accuracy: {final_acc:.2%}")
print("="*50)

# evaluate accuracy at different numbers of selected champions
print("\n--- Validation Breakdown by Draft Completeness Stage ---")
val_stages_x, val_stages_y = extract_champion_ids(df_val, augment_partial=True)
total_picks = (val_stages_x != num_champs).sum(axis=1)
val_stages_x_tensor = torch.tensor(val_stages_x, dtype=torch.long, device=device)

with torch.no_grad():
    val_stage_preds = (model(val_stages_x_tensor).cpu().numpy() > 0.5).astype(int).squeeze(-1)

stage_ranges = [
    ("Stage 1 (1-2 picks)", (1, 2)),
    ("Stage 2 (3-4 picks)", (3, 4)),
    ("Stage 3 (5-6 picks)", (5, 6)),
    ("Stage 4 (7-8 picks)", (7, 8)),
    ("Stage 5 (9-10 picks / Full)", (9, 10))
]

for name, (low, high) in stage_ranges:
    mask = (total_picks >= low) & (total_picks <= high)
    if mask.sum() > 0:
        acc = accuracy_score(val_stages_y[mask], val_stage_preds[mask])
        print(f"  {name:30s} : {acc:.2%} ({mask.sum()} samples)")

# Save model weights and metadata
torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)

metadata = {
    'champion_names': champs,
    'champ_to_idx': champ_to_idx,
    'accuracy': float(final_acc),
    'model_type': 'WideAndDeepDraftNN',
    'embedding_dim': EMBEDDING_DIM
}
with open(MODEL_META_PATH, 'w') as f:
    json.dump(metadata, f, indent=4)

# export model and lookup tables for browser inference
print(f"Exporting ONNX model to {ONNX_MODEL_PATH}...")
dummy_deep = torch.zeros((1, 10), dtype=torch.long, device=device)

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    torch.onnx.export(
        model,
        dummy_deep,
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=16,
        input_names=['x_deep'],
        output_names=['output'],
        dynamic_axes={
            'x_deep': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False
    )

print(f"Successfully exported PyTorch ONNX model ({os.path.getsize(ONNX_MODEL_PATH)/1024:.1f} KB) to ONNX_MODEL_PATH")

import os, json, sqlite3, torch, argparse, sys, pandas as pd, numpy as np, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)
from model import WideAndDeepDraftNN
from feature_engineering import build_feature_matrices, extract_draft_features

def load_data_from_db():
    conn = sqlite3.connect(os.path.join(ROOT_DIR, 'league_data.db'))
    df = pd.read_sql_query("SELECT * FROM matches", conn)
    conn.close()
    return df

def get_champion_metadata():
    with open(os.path.join(ROOT_DIR, 'champions.json'), 'r') as f:
        data = json.load(f)
    names = sorted(list(data.values())) if isinstance(data, dict) else sorted(list(data))
    return names, {name: i for i, name in enumerate(names)}

def vectorize_data(df, name_to_idx, feature_matrices, feature_scaler=None, augment_snake_draft=True):
    num_champs = len(name_to_idx)
    roles = ['top', 'jungle', 'mid', 'bot', 'support']
    snake_stages = [
        ([0], []), ([0], [0, 1]), ([0, 1, 2], [0, 1]),
        ([0, 1, 2], [0, 1, 2, 3]), ([0, 1, 2, 3, 4], [0, 1, 2, 3]),
        ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    ]
    stage_weights = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1.0], dtype=np.float32)

    blue_names = df[[f'blue_{r}' for r in roles]].to_numpy()
    red_names = df[[f'red_{r}' for r in roles]].to_numpy()
    blue_mat = np.array([[name_to_idx.get(v, num_champs) for v in r] for r in blue_names], dtype=np.int64)
    red_mat = np.array([[name_to_idx.get(v, num_champs) for v in r] for r in red_names], dtype=np.int64)
    wins = (df['winning_team'].to_numpy() == 'BLUE_WIN').astype(np.float32)

    num_matches = len(df)
    stages_to_use = list(enumerate(snake_stages)) if augment_snake_draft else [(5, snake_stages[-1])]
    total_samples = num_matches * len(stages_to_use)

    # Extract full Turn 6 raw features to establish non-distorted baseline scaler
    full_feats = np.array([extract_draft_features(b, r, feature_matrices) for b, r in zip(blue_names, red_names)], dtype=np.float32)
    if feature_scaler is None:
        mean = np.mean(full_feats, axis=0, keepdims=True)
        std = np.std(full_feats, axis=0, keepdims=True) + 1e-6
        feature_scaler = (mean, std)
    else:
        mean, std = feature_scaler

    raw_feats_stages = []
    for s_idx, (b_act, r_act) in stages_to_use:
        b_p = np.where(np.isin(np.arange(5), b_act)[None, :], blue_names, '')
        r_p = np.where(np.isin(np.arange(5), r_act)[None, :], red_names, '')
        stage_feats = np.array([extract_draft_features(b, r, feature_matrices) for b, r in zip(b_p, r_p)], dtype=np.float32)
        norm_stage = (stage_feats - mean) / std

        # Zero-mask per-slot features for unpicked/empty slots and team differentials on partial turns
        slot_mask = np.ones((len(df), 30), dtype=np.float32)
        for i in range(5):
            if i not in b_act:
                slot_mask[:, i * 3:(i + 1) * 3] = 0.0
            if i not in r_act:
                slot_mask[:, 15 + i * 3:15 + (i + 1) * 3] = 0.0

        norm_stage[:, :30] *= slot_mask
        if s_idx < 5:
            norm_stage[:, 30:] *= 0.0

        raw_feats_stages.append(norm_stage)

    X_feats = np.vstack(raw_feats_stages)
    X_deep = np.full((total_samples, 10), num_champs, dtype=np.int64)
    y_out = np.empty(total_samples, dtype=np.float32)
    w_out = np.empty(total_samples, dtype=np.float32)

    for idx, (s_idx, (b_act, r_act)) in enumerate(stages_to_use):
        slc = slice(idx * num_matches, (idx + 1) * num_matches)
        y_out[slc], w_out[slc] = wins, stage_weights[s_idx]
        for r in b_act:
            X_deep[slc, r] = blue_mat[:, r]
        for r in r_act:
            X_deep[slc, 5 + r] = red_mat[:, r]

    return X_deep, X_feats, y_out, w_out, feature_scaler

def train_and_evaluate(X_tr, F_tr, y_tr, w_tr, X_val, F_val, y_val, w_val, num_champs, embed_dim, num_heads, alpha, device, epochs=100, patience=10):
    d_tr, f_tr = torch.tensor(X_tr, dtype=torch.long, device=device), torch.tensor(F_tr, dtype=torch.float32, device=device)
    y_tr, w_tr = torch.tensor(y_tr, dtype=torch.float32, device=device).unsqueeze(1), torch.tensor(w_tr, dtype=torch.float32, device=device).unsqueeze(1)
    d_val, f_val = torch.tensor(X_val, dtype=torch.long, device=device), torch.tensor(F_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)

    model = WideAndDeepDraftNN(num_champs, embed_dim, num_heads, num_extra_features=F_tr.shape[1]).to(device)
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=alpha)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5)

    best_val_loss, best_state, patience_cnt, best_epoch = float('inf'), None, 0, 0
    N_tr = d_tr.size(0)
    N_val_matches = len(y_val) // 6 if len(y_val) % 6 == 0 else len(y_val)
    t6_slc = slice(5 * N_val_matches, 6 * N_val_matches) if len(y_val) % 6 == 0 else slice(0, len(y_val))

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N_tr, device=device)
        total_loss, total_w = 0.0, 0.0
        for i in range(0, N_tr, 2048):
            idx = perm[i:i + 2048]
            optimizer.zero_grad()
            loss = (criterion(model(d_tr[idx], f_tr[idx]), y_tr[idx]) * w_tr[idx]).sum()
            (loss / w_tr[idx].sum()).backward()
            optimizer.step()
            total_loss += loss.item()
            total_w += w_tr[idx].sum().item()

        tr_loss = total_loss / max(1.0, total_w)
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(d_val[t6_slc], f_val[t6_slc]), y_val_t[t6_slc]).mean().item()

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_cnt = 0
            best_epoch = epoch
            flag = " (Best)"
        else:
            patience_cnt += 1
            flag = ""

        if epoch % 5 == 0 or epoch == 1 or patience_cnt >= patience:
            print(f"  Epoch {epoch:3d}/{epochs} | Train Loss: {tr_loss:.4f} | Turn 6 Val Loss: {val_loss:.4f}{flag}")

        if patience_cnt >= patience:
            print(f"  Early stopping at Epoch {epoch}. Best Checkpoint: Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    stage_scores = {}
    with torch.no_grad():
        preds = (model(d_val, f_val).cpu().numpy() > 0.5).astype(int)
        names = ["Turn 1 (1 Pick)", "Turn 2 (3 Picks)", "Turn 3 (5 Picks)", "Turn 4 (7 Picks)", "Turn 5 (9 Picks)", "Turn 6 (Full 5v5 Draft)"]
        if len(y_val) % 6 == 0:
            for s_idx, s_name in enumerate(names):
                slc = slice(s_idx * N_val_matches, (s_idx + 1) * N_val_matches)
                stage_scores[s_name] = accuracy_score(y_val[slc], preds[slc])
        score = stage_scores.get("Turn 6 (Full 5v5 Draft)", accuracy_score(y_val, preds))

    return model, score, stage_scores

def save_files(model, champion_names, name_to_idx, score, stage_scores, embed_dim, num_heads, alpha, feature_scaler=None):
    meta_path = os.path.join(SCRIPT_DIR, 'model_nn_metadata.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                old_meta = json.load(f)
                old_acc = old_meta.get('accuracy', 0.0)
                if old_acc > score:
                    print(f"\n⚠️  Skipping save: New run accuracy ({score:.2%}) is lower than existing checkpoint ({old_acc:.2%}).")
                    print(f"    Preserved existing best model_nn.pth ({old_acc:.2%}).")
                    return
        except Exception:
            pass

    torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, 'model_nn.pth'))
    scaler_dict = {'mean': feature_scaler[0].tolist(), 'std': feature_scaler[1].tolist()} if feature_scaler else None
    metadata = {
        'champion_names': champion_names,
        'champ_to_idx': name_to_idx,
        'accuracy': score,
        'stage_accuracies': stage_scores,
        'model_type': 'PyTorch_WideAndDeepAttentionDraftNN',
        'embedding_dim': embed_dim,
        'num_heads': num_heads,
        'feature_scaler': scaler_dict,
        'best_hyperparameters': {'embedding_dim': embed_dim, 'num_heads': num_heads, 'alpha': alpha}
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"\n🏆 New Best Model Saved! Accuracy: {score:.2%}")
    for k, v in stage_scores.items():
        print(f"  - {k:25s}: {v:.2%}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1e-2)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    champ_names, name_to_idx = get_champion_metadata()
    cache_path = os.path.join(SCRIPT_DIR, 'dataset_cache.npz')

    if os.path.exists(cache_path) and not args.force:
        print(f"Loading cached vectorized dataset from {cache_path}...")
        cache = np.load(cache_path, allow_pickle=True)
        X_tr, F_tr, y_tr, w_tr = cache['X_tr'], cache['F_tr'], cache['y_tr'], cache['w_tr']
        X_val, F_val, y_val, w_val = cache['X_val'], cache['F_val'], cache['y_val'], cache['w_val']
        feature_scaler = cache['feature_scaler'].item()
    else:
        df = load_data_from_db()
        df_tr, df_val = train_test_split(df, test_size=0.2, random_state=42)
        feature_matrices = build_feature_matrices(df_tr)
        X_tr, F_tr, y_tr, w_tr, feature_scaler = vectorize_data(df_tr, name_to_idx, feature_matrices)
        X_val, F_val, y_val, w_val, _ = vectorize_data(df_val, name_to_idx, feature_matrices, feature_scaler=feature_scaler)
        np.savez_compressed(cache_path, X_tr=X_tr, F_tr=F_tr, y_tr=y_tr, w_tr=w_tr, X_val=X_val, F_val=F_val, y_val=y_val, w_val=w_val, feature_scaler=feature_scaler)
        print(f"Cached vectorized dataset to {cache_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Train: {len(y_tr)} | Val: {len(y_val)}")
    num_champs = len(champ_names)

    if args.tune:
        configs = [(8, 1), (8, 2), (8, 4), (16, 2), (16, 4)]
        best_acc, best_config, best_model, best_stages = 0.0, {}, None, {}
        for embed_dim, num_heads in configs:
            print(f"\n--- Testing EmbedDim={embed_dim}, Heads={num_heads}, Alpha={args.alpha} ---")
            m, s, st = train_and_evaluate(X_tr, F_tr, y_tr, w_tr, X_val, F_val, y_val, w_val, num_champs, embed_dim, num_heads, args.alpha, device, patience=10)
            t6_acc = st.get("Turn 6 (Full 5v5 Draft)", s)
            print(f" -> Turn 6 Val Accuracy: {t6_acc:.2%} | Blended Acc: {s:.2%}")
            if s > best_acc:
                best_acc = s
                best_config = {'embedding_dim': embed_dim, 'num_heads': num_heads, 'alpha': args.alpha}
                best_model = m
                best_stages = st
        save_files(best_model, champ_names, name_to_idx, best_acc, best_stages, best_config['embedding_dim'], best_config['num_heads'], best_config['alpha'], feature_scaler)
    else:
        m, s, st = train_and_evaluate(X_tr, F_tr, y_tr, w_tr, X_val, F_val, y_val, w_val, num_champs, args.embed_dim, 2, args.alpha, device, patience=10)
        save_files(m, champ_names, name_to_idx, s, st, args.embed_dim, 2, args.alpha, feature_scaler)

if __name__ == "__main__":
    main()

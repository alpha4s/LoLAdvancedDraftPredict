import torch, torch.nn as nn
from feature_engineering import NUM_SLOT_FEATS_PER_CHAMP, NUM_SLOT_TOTAL_FEATS

class WideAndDeepDraftNN(nn.Module):
    def __init__(self, num_champs, embedding_dim=16, num_heads=2, num_extra_features=32):
        super().__init__()
        self.num_champs = num_champs
        self.wide_bias = nn.Embedding(num_champs * 5, 1)
        nn.init.zeros_(self.wide_bias.weight)
        self.register_buffer('slot_signs', torch.tensor([1.0]*5 + [-1.0]*5, dtype=torch.float32))
        self.register_buffer('role_indices', torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long))

        self.champ_embeddings = nn.Embedding(num_champs + 1, embedding_dim, padding_idx=num_champs)
        self.role_embeddings = nn.Embedding(5, embedding_dim)
        self.slot_proj = nn.Linear(embedding_dim + NUM_SLOT_FEATS_PER_CHAMP, embedding_dim)
        nn.init.normal_(self.slot_proj.weight, std=0.01)
        nn.init.zeros_(self.slot_proj.bias)

        num_team_features = max(0, num_extra_features - NUM_SLOT_TOTAL_FEATS)
        fc_input_dim = 3 * embedding_dim + num_team_features
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(fc_input_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_deep, x_features=None):
        bsz = x_deep.size(0)
        padding_mask = (x_deep == self.num_champs)
        role_idx = self.role_indices.unsqueeze(0).expand(bsz, -1)

        flat_idx = role_idx * self.num_champs + torch.clamp(x_deep, 0, self.num_champs - 1)
        wide_out = (self.wide_bias(flat_idx).squeeze(-1) * self.slot_signs * (~padding_mask).float()).sum(dim=1, keepdim=True)

        sequence = self.champ_embeddings(x_deep) + self.role_embeddings(role_idx)
        if x_features is not None and x_features.shape[1] >= NUM_SLOT_TOTAL_FEATS:
            slot_feats = x_features[:, :NUM_SLOT_TOTAL_FEATS].view(bsz, 10, NUM_SLOT_FEATS_PER_CHAMP)
            sequence = self.slot_proj(torch.cat([sequence, slot_feats], dim=-1))
            team_feats = x_features[:, NUM_SLOT_TOTAL_FEATS:]
        else:
            team_feats = x_features

        seq_norm = self.ln1(sequence)
        attn_out, _ = self.attn(seq_norm, seq_norm, seq_norm, key_padding_mask=padding_mask)
        sequence = sequence + torch.nan_to_num(attn_out, nan=0.0)

        b_mask, r_mask = (~padding_mask[:, :5]).unsqueeze(-1).float(), (~padding_mask[:, 5:]).unsqueeze(-1).float()
        blue_rep = (sequence[:, :5] * b_mask).sum(dim=1) / b_mask.sum(dim=1).clamp(min=1.0)
        red_rep = (sequence[:, 5:] * r_mask).sum(dim=1) / r_mask.sum(dim=1).clamp(min=1.0)
        cross_rep = blue_rep * red_rep

        combined = torch.cat([blue_rep, red_rep, cross_rep, team_feats], dim=1) if team_feats is not None and team_feats.shape[1] > 0 else torch.cat([blue_rep, red_rep, cross_rep], dim=1)
        return torch.sigmoid(wide_out + self.fc(self.ln2(combined)))

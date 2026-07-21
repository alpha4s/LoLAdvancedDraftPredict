import torch
import torch.nn as nn

class WideAndDeepDraftNN(nn.Module):
    def __init__(self, num_champs, embedding_dim=16, num_heads=2):
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

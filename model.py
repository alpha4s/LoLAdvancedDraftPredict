import torch
import torch.nn as nn
import torch.nn.functional as F

class WideAndDeepDraftNN(nn.Module):
    """
    Graph Neural Network + Wide & Deep Transformer Architecture for League Draft Prediction.
    - GCN Layer: Blends champion representations with pairwise synergy & counter graph topology.
    - Self-Attention: Learns draft-wide composition interactions.
    - Masking Support: Handles partial drafts cleanly via key_padding_mask and masked mean pooling.
    """
    def __init__(self, num_champs, embedding_dim=16, num_heads=2, adj_matrix=None):
        super(WideAndDeepDraftNN, self).__init__()
        self.num_champs = num_champs
        self.embedding_dim = embedding_dim
        
        self.wide_linear = nn.Linear(num_champs * 5, 1, bias=False)
        self.champ_embeddings = nn.Embedding(num_champs + 1, embedding_dim, padding_idx=num_champs)
        self.role_embeddings = nn.Embedding(5, embedding_dim)
        
        # Graph Convolution Layer
        self.gnn_linear = nn.Linear(embedding_dim, embedding_dim)
        if adj_matrix is not None:
            if not isinstance(adj_matrix, torch.Tensor):
                adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            # Normalize adjacency matrix D^(-1/2) A D^(-1/2)
            deg = adj_matrix.sum(dim=1, keepdim=True).clamp(min=1.0)
            norm_adj = adj_matrix / deg
            self.register_buffer('norm_adj', norm_adj)
        else:
            self.register_buffer('norm_adj', None)

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True, dropout=0.2)
        self.ln2 = nn.LayerNorm(2 * embedding_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(2 * embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 1)
        )
        
    def get_champion_representations(self):
        """Applies GCN message passing over champion embeddings using the graph topology."""
        embeds = self.champ_embeddings.weight # (num_champs+1, dim)
        if hasattr(self, 'norm_adj') and self.norm_adj is not None:
            # GCN: H' = ReLU(NormAdj * H * W_g)
            gnn_out = F.relu(self.gnn_linear(torch.matmul(self.norm_adj, embeds)))
            return embeds + gnn_out # Residual connection
        return embeds

    def forward(self, x_wide, x_deep):
        wide_out = self.wide_linear(x_wide)
        
        batch_size = x_deep.size(0)
        device = x_deep.device
        
        # Get graph-refined champion embeddings
        all_embeds = self.get_champion_representations()
        champs = F.embedding(x_deep, all_embeds, padding_idx=self.num_champs)
        
        role_idx = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long, device=device)
        role_idx = role_idx.unsqueeze(0).expand(batch_size, -1)
        roles = self.role_embeddings(role_idx)
        
        seq = champs + roles
        
        # Key padding mask for empty slots (token == num_champs)
        key_padding_mask = (x_deep == self.num_champs) # (batch_size, 10)
        
        # Self-Attention over active picks
        attn_out, _ = self.attn(seq, seq, seq, key_padding_mask=key_padding_mask)
        # If all items in sequence are padded, replace NaNs with zeros
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        seq = self.ln1(seq + attn_out)
        
        # Masked mean pooling for Blue (first 5) and Red (last 5)
        blue_mask = (~key_padding_mask[:, :5]).unsqueeze(-1).float() # (batch, 5, 1)
        red_mask = (~key_padding_mask[:, 5:]).unsqueeze(-1).float()   # (batch, 5, 1)
        
        blue_sum = (seq[:, :5, :] * blue_mask).sum(dim=1)
        blue_cnt = blue_mask.sum(dim=1).clamp(min=1.0)
        blue_rep = blue_sum / blue_cnt
        
        red_sum = (seq[:, 5:, :] * red_mask).sum(dim=1)
        red_cnt = red_mask.sum(dim=1).clamp(min=1.0)
        red_rep = red_sum / red_cnt
        
        combined = torch.cat([blue_rep, red_rep], dim=1)
        combined = self.ln2(combined)
        
        deep_out = self.fc(combined)
        return torch.sigmoid(wide_out + deep_out)

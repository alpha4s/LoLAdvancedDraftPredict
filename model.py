import torch
import torch.nn as nn
from config import EMBEDDING_DIM, NUM_EXTRA_FEATURES

class WideAndDeepDraftNN(nn.Module):
    """
    Wide-and-deep model for predicting the Blue-side win probability.
    The exported model includes the feature lookup tables, so inference only
    needs ten champion IDs: five Blue-side picks followed by five Red-side picks.
    """
    def __init__(self, num_champs, features, embed_dim=EMBEDDING_DIM):
        super().__init__()
        self.num_champs = num_champs

        # init champ vecs
        self.champ_embed = nn.Embedding(num_champs + 1, embed_dim, padding_idx=num_champs)
        nn.init.normal_(self.champ_embed.weight, std=0.01)

        from_pretrained = lambda name: nn.Embedding.from_pretrained(features[name], freeze=True)

        # feature engineering champ vecs [num_champs+1, 5]
        self.ap_ratios = from_pretrained('ap_ratios')
        self.ap_variances = from_pretrained('ap_variances')
        self.role_freqs = from_pretrained('role_freqs')

        # matchup champ vecs [(num_champs+1)^2, 1]
        self.top_counters = from_pretrained('top_counters')
        self.mid_counters = from_pretrained('mid_counters')
        self.supp_counters = from_pretrained('supp_counters') 

        num_extra_features = NUM_EXTRA_FEATURES

        # per champ role differentiation [num_champs+1, 5] WIDE PATH
        self.wide_champ_roles = nn.Embedding(num_champs + 1, 5, padding_idx=num_champs) 
        nn.init.zeros_(self.wide_champ_roles.weight)
        self.wide = nn.Linear(num_extra_features, 1)
        nn.init.zeros_(self.wide.weight)
        nn.init.zeros_(self.wide.bias)

        # MLP. Champ vec + 44 engineered features. DEEP PATH
        deep_in_dim = (10 * embed_dim) + num_extra_features
        self.deep = nn.Sequential(
            nn.Linear(deep_in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def extract_graph_features(self, x_deep):
        bsz = x_deep.size(0)
        
        b_ids = x_deep[:, :5]
        r_ids = x_deep[:, 5:]
        b_ap = self.ap_ratios(b_ids).squeeze(-1)
        b_var = self.ap_variances(b_ids).squeeze(-1)
        r_ap = self.ap_ratios(r_ids).squeeze(-1)
        r_var = self.ap_variances(r_ids).squeeze(-1)

        b_rf = self.role_freqs(b_ids).diagonal(dim1=1, dim2=2)
        r_rf = self.role_freqs(r_ids).diagonal(dim1=1, dim2=2)
        wide_b = self.wide_champ_roles(b_ids).diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True)
        wide_r = self.wide_champ_roles(r_ids).diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True)

        b_off = (0.2 - b_rf).clamp(min=0.0) * (b_ids != self.num_champs).float()
        r_off = (0.2 - r_rf).clamp(min=0.0) * (r_ids != self.num_champs).float()

        b_slot_feats = torch.stack([b_ap, b_var, b_rf, b_off], dim=-1).view(bsz, 20)
        r_slot_feats = torch.stack([r_ap, r_var, r_rf, r_off], dim=-1).view(bsz, 20)

        stride = self.num_champs + 1
        b_top = self.top_counters(b_ids[:, 0] * stride + r_ids[:, 0]).squeeze(-1)
        r_top = self.top_counters(r_ids[:, 0] * stride + b_ids[:, 0]).squeeze(-1)

        b_mid = self.mid_counters(b_ids[:, 2] * stride + r_ids[:, 2]).squeeze(-1)
        r_mid = self.mid_counters(r_ids[:, 2] * stride + b_ids[:, 2]).squeeze(-1)

        b_supp = self.supp_counters(b_ids[:, 4] * stride + r_ids[:, 4]).squeeze(-1)
        r_supp = self.supp_counters(r_ids[:, 4] * stride + b_ids[:, 4]).squeeze(-1)

        lane_counters = torch.stack([b_top - r_top, b_mid - r_mid, b_supp - r_supp], dim=1)

        b_ap_total = b_ap.sum(dim=1, keepdim=True)
        r_ap_total = r_ap.sum(dim=1, keepdim=True)
        ap_balance_delta = b_ap_total - r_ap_total

        x_features = torch.cat([b_slot_feats, r_slot_feats, lane_counters, ap_balance_delta], dim=1)
        return x_features, wide_b, wide_r

    def forward(self, x_deep):
        bsz = x_deep.size(0)
        embeds = self.champ_embed(x_deep).view(bsz, -1)
        x_features, wide_b, wide_r = self.extract_graph_features(x_deep)

        wide_out = self.wide(x_features) + (wide_b - wide_r)

        deep_in = torch.cat([embeds, x_features], dim=1)
        deep_out = self.deep(deep_in)
        return torch.sigmoid(wide_out + deep_out)

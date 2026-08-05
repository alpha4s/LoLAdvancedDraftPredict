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

        # feature engineering champ vecs [num_champs+1, 5]
        self.ap_ratios = nn.Embedding.from_pretrained(features['ap_ratios'], freeze=True)
        self.ap_variances = nn.Embedding.from_pretrained(features['ap_variances'], freeze=True)
        self.role_freqs = nn.Embedding.from_pretrained(features['role_freqs'], freeze=True) 

        # matchup champ vecs [(num_champs+1)^2, 1]
        self.top_counters = nn.Embedding.from_pretrained(features['top_counters'], freeze=True) 
        self.mid_counters = nn.Embedding.from_pretrained(features['mid_counters'], freeze=True)   
        self.supp_counters = nn.Embedding.from_pretrained(features['supp_counters'], freeze=True) 

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
        
        # [batch, 5]
        blue_ids = x_deep[:, :5]
        red_ids = x_deep[:, 5:]
        blue_ap = self.ap_ratios(blue_ids).squeeze(-1)
        blue_var = self.ap_variances(blue_ids).squeeze(-1)
        red_ap = self.ap_ratios(red_ids).squeeze(-1)
        red_var = self.ap_variances(red_ids).squeeze(-1)

        # sets easy to work with data lists
        blue_rf_cols = []
        red_rf_cols = []
        wide_blue_cols = []
        wide_red_cols = []
        for i in range(5):
            blue_rf_cols.append(self.role_freqs(blue_ids[:, i])[:, i])
            red_rf_cols.append(self.role_freqs(red_ids[:, i])[:, i])
            wide_blue_cols.append(self.wide_champ_roles(blue_ids[:, i])[:, i])
            wide_red_cols.append(self.wide_champ_roles(red_ids[:, i])[:, i])

        # turn lists into tensors
        blue_rf = torch.stack(blue_rf_cols, dim=1)
        red_rf = torch.stack(red_rf_cols, dim=1)
        wide_blue = torch.stack(wide_blue_cols, dim=1).sum(dim=1, keepdim=True)
        wide_red = torch.stack(wide_red_cols, dim=1).sum(dim=1, keepdim=True)

        # off meta penalty
        blue_off = (0.2 - blue_rf).clamp(min=0.0) * (blue_ids != self.num_champs).float()
        red_off = (0.2 - red_rf).clamp(min=0.0) * (red_ids != self.num_champs).float()

        blue_slot_feats = torch.stack([blue_ap, blue_var, blue_rf, blue_off], dim=-1).view(bsz, 20)
        red_slot_feats = torch.stack([red_ap, red_var, red_rf, red_off], dim=-1).view(bsz, 20)

        # look up top, mid, supp matchups
        stride = self.num_champs + 1

        b_top = self.top_counters(blue_ids[:, 0] * stride + red_ids[:, 0]).squeeze(-1)
        r_top = self.top_counters(red_ids[:, 0] * stride + blue_ids[:, 0]).squeeze(-1)

        b_mid = self.mid_counters(blue_ids[:, 2] * stride + red_ids[:, 2]).squeeze(-1)
        r_mid = self.mid_counters(red_ids[:, 2] * stride + blue_ids[:, 2]).squeeze(-1)

        b_supp = self.supp_counters(blue_ids[:, 4] * stride + red_ids[:, 4]).squeeze(-1)
        r_supp = self.supp_counters(red_ids[:, 4] * stride + blue_ids[:, 4]).squeeze(-1)

        lane_counters = torch.stack([b_top - r_top, b_mid - r_mid, b_supp - r_supp], dim=1)

        b_ap_total = blue_ap.sum(dim=1, keepdim=True)
        r_ap_total = red_ap.sum(dim=1, keepdim=True)
        ap_balance_delta = b_ap_total - r_ap_total

        x_features = torch.cat([blue_slot_feats, red_slot_feats, lane_counters, ap_balance_delta], dim=1)
        return x_features, wide_blue, wide_red

    def forward(self, x_deep):
        bsz = x_deep.size(0)
        embeds = self.champ_embed(x_deep).view(bsz, -1)
        x_features, wide_blue, wide_red = self.extract_graph_features(x_deep)

        wide_out = self.wide(x_features) + (wide_blue - wide_red)

        deep_in = torch.cat([embeds, x_features], dim=1)
        deep_out = self.deep(deep_in)
        return torch.sigmoid(wide_out + deep_out)

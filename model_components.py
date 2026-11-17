import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransLayer(nn.Module):
    """
    Temporal-Weather-Aware Translation Layer.
    Calculates the distance g(l_i, r_i, l_j) = || e_li + e_ri - e_lj ||^2
    """
    def __init__(self):
        super(TransLayer, self).__init__()

    def forward(self, poi_emb_head, relation_emb, poi_emb_tail):
        """
        Calculates the distance g(l_i, r_i, l_j) = || e_li + e_ri - e_lj ||^2
        Requires relation_emb to be pre-combined (e.g., Time + Weather).
        """
        # diff = h + r - t
        diff = poi_emb_head + relation_emb - poi_emb_tail
        
        # Distance (L2 norm squared)
        distance = torch.sum(diff ** 2, dim=1)
        return distance

    def predict_likelihood(self, poi_emb_head, relation_emb, poi_emb_tail):
        """
        Equation (3): s(...) = exp(-g(...))
        """
        distance = self.forward(poi_emb_head, relation_emb, poi_emb_tail)
        return torch.exp(-distance)

    def get_top_k_neighbors(self, poi_emb_head, relation_emb, all_poi_embs, k=5):
        """
        Select top k POIs as adjacent nodes based on likelihood.
        """
        # query = h + r
        query = poi_emb_head + relation_emb # [batch_size, dim]
        
        # Calculate distance to ALL POIs
        # [batch_size, num_pois]
        dists = torch.cdist(query, all_poi_embs, p=2)
        
        # Top K (Smallest distance = Highest likelihood)
        _, indices = dists.topk(k, largest=False)
        
        return indices


class UserGraphLayer(nn.Module):
    """
    One layer of Message Passing for User-POI Interaction Graph (Eq 4-7).
    """
    def __init__(self, embed_dim):
        super(UserGraphLayer, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, 
                user_emb, poi_emb, time_emb, weather_emb,
                edges,
                season_emb=None, day_weather_emb=None, month_weather_emb=None,
                e_W_per_edge=None, use_season=False, is_baseline=False,
                skip_user_agg=False, use_time=True, use_weather=True):
        """
        Unified implementation of User-POI GNN layers.
        """
        u_idx = edges['user_idxs']
        p_idx = edges['poi_idxs']
        t_idx = edges['time_idxs']
        # アブレーション用: 時間 / 単一の気象情報をメッセージに含めない
        e_t = time_emb[t_idx] if use_time else 0
        
        if is_baseline:
            # Baseline uses 'weather_idxs'
            w_idx = edges['weather_idxs'] if 'weather_idxs' in edges else edges['weather_now_idxs']
            # Message: e_p + e_t + e_w
            e_w = weather_emb[w_idx] if use_weather else 0
            msg_for_user = poi_emb[p_idx] + e_t + e_w
            msg_for_poi = user_emb[u_idx] + e_t + e_w
        else:
            # Proposed: multi-granular weather & season
            s_idx = edges.get('season_idxs')
            w_now_idx = edges.get('weather_now_idxs')
            w_day_idx = edges.get('weather_day_idxs')
            w_month_idx = edges.get('weather_month_idxs')
            
            # Message using pre-calculated e_W (Weather Context)
            if use_season and season_emb is not None:
                msg_for_user = poi_emb[p_idx] + e_t + season_emb[s_idx] + e_W_per_edge
                msg_for_poi = user_emb[u_idx] + e_t + season_emb[s_idx] + e_W_per_edge
            else:
                msg_for_user = poi_emb[p_idx] + e_t + e_W_per_edge
                msg_for_poi = user_emb[u_idx] + e_t + e_W_per_edge

        # 1. Update User Embeddings
        new_user_emb = torch.zeros_like(user_emb)
        user_counts = torch.zeros(user_emb.size(0), 1, device=user_emb.device)
        new_user_emb.index_add_(0, u_idx, msg_for_user)
        user_counts.index_add_(0, u_idx, torch.ones((u_idx.size(0), 1), device=user_counts.device))
        new_user_emb = new_user_emb / (user_counts + 1e-9)
        
        # 2. Update POI Embeddings
        new_poi_emb = torch.zeros_like(poi_emb)
        poi_counts = torch.zeros(poi_emb.size(0), 1, device=poi_emb.device)
        new_poi_emb.index_add_(0, p_idx, msg_for_poi)
        poi_counts.index_add_(0, p_idx, torch.ones((p_idx.size(0), 1), device=poi_counts.device))
        new_poi_emb = new_poi_emb / (poi_counts + 1e-9)

        # 3. Update Time Embeddings
        msg_for_time = user_emb[u_idx] + poi_emb[p_idx]
        new_time_emb = torch.zeros_like(time_emb)
        time_counts = torch.zeros(time_emb.size(0), 1, device=time_emb.device)
        new_time_emb.index_add_(0, t_idx, msg_for_time)
        time_counts.index_add_(0, t_idx, torch.ones((t_idx.size(0), 1), device=time_counts.device))
        new_time_emb = new_time_emb / (time_counts + 1e-9)

        # 4. Update Weather Embeddings (Now Weather or Single Weather)
        msg_for_weather = user_emb[u_idx] + poi_emb[p_idx]
        new_weather_emb = torch.zeros_like(weather_emb)
        weather_counts = torch.zeros(weather_emb.size(0), 1, device=weather_emb.device)
        
        target_w_idx = edges['weather_idxs'] if is_baseline and 'weather_idxs' in edges else edges['weather_now_idxs']
        new_weather_emb.index_add_(0, target_w_idx, msg_for_weather)
        weather_counts.index_add_(0, target_w_idx, torch.ones((target_w_idx.size(0), 1), device=weather_counts.device))
        new_weather_emb = new_weather_emb / (weather_counts + 1e-9)

        # Layer Combination (LightGCN, K=1): average ego (layer 0) and propagated (layer 1) embeddings
        new_user_emb = (user_emb + new_user_emb) / 2
        new_poi_emb = (poi_emb + new_poi_emb) / 2
        new_time_emb = (time_emb + new_time_emb) / 2
        new_weather_emb = (weather_emb + new_weather_emb) / 2

        # Apply Aggregation (Neighborhood Smoothing)
        if not skip_user_agg:
            new_user_emb = self.aggregation(new_user_emb, k=5)
        
        if is_baseline:
            return new_user_emb, new_poi_emb, new_time_emb, new_weather_emb
        
        # 5. Update Proposed Extra Embeddings
        new_season_emb = None
        if use_season and season_emb is not None:
            new_season_emb = torch.zeros_like(season_emb)
            season_counts = torch.zeros(season_emb.size(0), 1, device=season_emb.device)
            new_season_emb.index_add_(0, edges['season_idxs'], msg_for_weather) # Use common msg
            season_counts.index_add_(0, edges['season_idxs'], torch.ones((edges['season_idxs'].size(0), 1), device=season_counts.device))
            new_season_emb = (season_emb + new_season_emb / (season_counts + 1e-9)) / 2
        
        # 6. Update Day Weather Embeddings
        # w_day_idx: [num_edges, 3]. Flatten to update.
        msg_for_day = user_emb[u_idx] + poi_emb[p_idx] # Context
        
        # Repeat context for each of the 3 days
        msg_expanded = msg_for_day.repeat_interleave(3, dim=0) # [3*num_edges, dim]
        day_idx_expanded = w_day_idx.view(-1) # [3*num_edges]
        
        new_day_emb = torch.zeros_like(day_weather_emb)
        day_counts = torch.zeros(day_weather_emb.size(0), 1, device=day_weather_emb.device)
        
        new_day_emb.index_add_(0, day_idx_expanded, msg_expanded)
        day_counts.index_add_(0, day_idx_expanded, torch.ones((day_idx_expanded.size(0), 1), device=day_counts.device))
        new_day_emb = (day_weather_emb + new_day_emb / (day_counts + 1e-9)) / 2
            
        new_month_emb = None
        if month_weather_emb is not None:
            new_month_emb = torch.zeros_like(month_weather_emb)
            month_counts = torch.zeros(month_weather_emb.size(0), 1, device=month_weather_emb.device)
            new_month_emb.index_add_(0, w_month_idx, msg_for_weather)
            month_counts.index_add_(0, w_month_idx, torch.ones((w_month_idx.size(0), 1), device=month_counts.device))
            new_month_emb = (month_weather_emb + new_month_emb / (month_counts + 1e-9)) / 2

        return new_user_emb, new_poi_emb, new_time_emb, new_weather_emb, new_season_emb, new_day_emb, new_month_emb

    def aggregation(self, user_emb, k=5):
        # Optimization: Only aggregate if k > 0
        if k <= 0: return user_emb
        dist = torch.cdist(user_emb, user_emb, p=2)
        _, indices = dist.topk(k=k+1, largest=False)
        neighbor_embs = user_emb[indices]
        avg_emb = neighbor_embs.mean(dim=1)
        return avg_emb


def inv_softplus(y):
    """softplus(x) = y となる x（正値パラメータの初期化用）"""
    return math.log(math.expm1(y))


class SpatialTemporalSelfAttention(nn.Module):
    """
    Self-Attention with Spatial and Temporal distance penalty.
        score_ij = Q_i K_j^T / sqrt(d) - w_s * f(d_s(i,j)) - w_t * f(d_t(i,j))

    dist_norm:
      'legacy' : f(d) = log(1+d)（単位は km / h のまま、w は制約なし）
      'log'    : f(d) = min( log(1+d) / log(1+D), 1 ) ∈ [0,1]、w = softplus(θ) > 0
                 D は学習データの軌跡内距離・時間差のパーセンタイル（set_dist_scale で設定）
    """
    def __init__(self, embed_dim, dropout=0.1, dist_norm='legacy'):
        super(SpatialTemporalSelfAttention, self).__init__()
        assert dist_norm in ('legacy', 'log')
        self.embed_dim = embed_dim
        self.dist_norm = dist_norm
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        # 初期値はどちらも w = 1.0（log では softplus(θ) = 1.0 となる θ）
        init = 1.0 if dist_norm == 'legacy' else inv_softplus(1.0)
        self.w_s_raw = nn.Parameter(torch.tensor(init))
        self.w_t_raw = nn.Parameter(torch.tensor(init))
        if dist_norm == 'log':
            self.register_buffer('d_s_max', torch.tensor(1.0))
            self.register_buffer('d_t_max', torch.tensor(1.0))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # 旧チェックポイント（パラメータ名 w_s / w_t）との互換
        for name in ('w_s', 'w_t'):
            if prefix + name in state_dict and prefix + name + '_raw' not in state_dict:
                state_dict[prefix + name + '_raw'] = state_dict.pop(prefix + name)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def w_s(self):
        return self.w_s_raw if self.dist_norm == 'legacy' else F.softplus(self.w_s_raw)

    @property
    def w_t(self):
        return self.w_t_raw if self.dist_norm == 'legacy' else F.softplus(self.w_t_raw)

    def set_dist_scale(self, d_s_max, d_t_max):
        if self.dist_norm == 'legacy': return
        self.d_s_max.fill_(float(d_s_max)); self.d_t_max.fill_(float(d_t_max))

    def _penalty_feature(self, d, d_max):
        if self.dist_norm == 'legacy':
            return torch.log(d + 1.0)
        return (torch.log1p(d) / torch.log1p(d_max)).clamp(max=1.0)

    def spatial_feature(self, d_s):
        """空間距離 [km] → ペナルティ用の特徴量 f(d_s)"""
        return self._penalty_feature(d_s, getattr(self, 'd_s_max', None))

    def temporal_feature(self, d_t):
        """時間間隔 [h] → ペナルティ用の特徴量 f(d_t)"""
        return self._penalty_feature(d_t, getattr(self, 'd_t_max', None))

    def forward(self, trajectory_emb, abs_time_seq, lat_seq, lon_seq, mask=None):
        """abs_time_seq: 絶対時間 [h]（時刻カテゴリではない）"""
        Q = self.W_Q(trajectory_emb)
        K = self.W_K(trajectory_emb)
        V = self.W_V(trajectory_emb)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        # Time interval [h]: |t_j - t_i|
        t_i = abs_time_seq.unsqueeze(2)
        t_j = abs_time_seq.unsqueeze(1)
        d_t = torch.abs(t_j - t_i).float()

        # Spatial distance: Haversine [km]
        lat_i, lon_i = lat_seq.unsqueeze(2), lon_seq.unsqueeze(2)
        lat_j, lon_j = lat_seq.unsqueeze(1), lon_seq.unsqueeze(1)
        d_s = self.tensor_haversine(lat_i, lon_i, lat_j, lon_j)

        # 時空間的な距離が大きいほどアテンション（重み）は小さくなる
        scores = scores - self.w_s * self.spatial_feature(d_s) - self.w_t * self.temporal_feature(d_t)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output

    def tensor_haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = torch.deg2rad(lat2 - lat1)
        dlon = torch.deg2rad(lon2 - lon1)
        a = torch.sin(dlat / 2)**2 + torch.cos(torch.deg2rad(lat1)) * torch.cos(torch.deg2rad(lat2)) * torch.sin(dlon / 2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        return R * c

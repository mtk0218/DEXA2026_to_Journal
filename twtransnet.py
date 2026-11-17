import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_components import TransLayer, UserGraphLayer, SpatialTemporalSelfAttention

class TWTransNet(nn.Module):
    def __init__(self, num_users, num_pois, embed_dim=50, num_times=24, num_now_weathers=8, 
                 num_day_weathers=4, num_month_weathers=9, dropout=0.1, gamma=1.0, 
                 use_season=True, use_gnn=True, is_baseline=False, skip_user_agg=False, skip_poi_agg=False,
                 use_enhanced_traj=False, enhanced_traj_k=6, use_time=True, use_weather=True, seq_len=10, dist_norm='legacy'):
        super(TWTransNet, self).__init__()
        self.num_users = num_users
        self.num_pois = num_pois
        self.embed_dim = embed_dim
        self.gamma = gamma
        self.is_baseline = is_baseline
        self.use_season = use_season if not is_baseline else False
        self.use_gnn = use_gnn
        # アブレーション用: False の場合、時間情報 / 単一の気象情報(weather_now)の寄与を0にする
        self.use_time = use_time
        self.use_weather = use_weather
        self.skip_user_agg = skip_user_agg
        self.skip_poi_agg = skip_poi_agg
        self.use_enhanced_traj = use_enhanced_traj
        self.enhanced_traj_k = enhanced_traj_k
        self.seq_len = seq_len
        # POI遷移グラフ G_T の後続POI表（set_transition_neighbors で設定。state_dict には含めない）
        self.trans_nbr_idx, self.trans_nbr_mask = None, None

        # 1. Embeddings
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.poi_embedding = nn.Embedding(num_pois, embed_dim)
        self.time_embedding = nn.Embedding(num_times, embed_dim)
        
        # Weather Embedding: Single granularity for baseline, primary granularity for proposed
        self.weather_now_embedding = nn.Embedding(num_now_weathers, embed_dim)
        
        if not self.is_baseline:
            # Proposed Extra Contexts
            self.day_weather_embedding = nn.Embedding(num_day_weathers, embed_dim)
            self.month_weather_embedding = nn.Embedding(num_month_weathers, embed_dim)
            if self.use_season:
                self.season_embedding = nn.Embedding(4, embed_dim)
            else:
                self.season_embedding = None
            
            # Context Aggregation Parameters
            self.W_day_agg = nn.Linear(3 * embed_dim, embed_dim)
            self.b_d = nn.Parameter(torch.zeros(embed_dim))
            self.W_weather_agg = nn.Linear(3 * embed_dim, embed_dim)
            self.b_w = nn.Parameter(torch.zeros(embed_dim))
        else:
            # Undefined weights for baseline
            self.day_weather_embedding = None
            self.month_weather_embedding = None
            self.season_embedding = None
            self.W_day_agg = None
            self.W_weather_agg = None

        # Eq (8) POI Aggregation Parameters (Common)
        self.W_poi_agg = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.ReLU()

        # CAAI Eq (8): 強化軌跡グラフの近傍集約 e_li = σ(1/K Σ W1 e_l)
        self.W_enh = nn.Linear(embed_dim, embed_dim, bias=False) if use_enhanced_traj else None
        # 予測層の系列位置ごとの学習重み W2 ∈ R^{n}（STAN の Linear(M→1) に相当、softmax なし）。
        # 1 初期化 = 有効位置の単純平均から学習を始める（predict 参照）
        self.W2 = nn.Parameter(torch.ones(seq_len))

        # 2. Modules
        self.trans_layer = TransLayer()
        self.gnn_layer = UserGraphLayer(embed_dim)
        self.st_attention = SpatialTemporalSelfAttention(embed_dim, dropout=dropout, dist_norm=dist_norm)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.poi_embedding.weight)
        nn.init.xavier_uniform_(self.time_embedding.weight)
        nn.init.xavier_uniform_(self.weather_now_embedding.weight)
        if not self.is_baseline:
            nn.init.xavier_uniform_(self.day_weather_embedding.weight)
            nn.init.xavier_uniform_(self.month_weather_embedding.weight)
            if self.use_season:
                nn.init.xavier_uniform_(self.season_embedding.weight)

    def forward_gnn(self, edges):
        u_emb = self.user_embedding.weight
        p_emb = self.poi_embedding.weight
        t_emb = self.time_embedding.weight
        w_now_emb = self.weather_now_embedding.weight

        if self.is_baseline:
            # Baseline MP: Always run GNN for existing method
            return self.gnn_layer(u_emb, p_emb, t_emb, w_now_emb, edges, is_baseline=True, skip_user_agg=self.skip_user_agg,
                                  use_time=self.use_time, use_weather=self.use_weather)

        if not self.use_gnn:
            # Proposed method: Return original weights if GNN is disabled
            s_emb = self.season_embedding.weight if self.use_season else None
            d_emb = self.day_weather_embedding.weight
            m_emb = self.month_weather_embedding.weight
            return u_emb, p_emb, t_emb, w_now_emb, s_emb, d_emb, m_emb

        # Proposed method: Build multi-granular context first and run GNN
        s_emb = self.season_embedding.weight if self.use_season else None
        d_emb = self.day_weather_embedding.weight
        m_emb = self.month_weather_embedding.weight

        # Pre-compute e_W for message passing
        e_now = w_now_emb[edges['weather_now_idxs']]
        if not self.use_weather: e_now = torch.zeros_like(e_now)
        w_day_seq_idx = edges['weather_day_idxs']
        w_yesterday = d_emb[w_day_seq_idx[:, 0]]
        w_today = d_emb[w_day_seq_idx[:, 1]]
        w_tomorrow = d_emb[w_day_seq_idx[:, 2]]
        
        concat_day = torch.cat([w_yesterday, w_today, w_tomorrow], dim=1)
        e_day = torch.matmul(concat_day, self.W_day_agg.weight.t()) + self.b_d
        
        e_month = m_emb[edges['weather_month_idxs']]
        concat_weather = torch.cat([e_now, e_day, e_month], dim=1)
        e_W = torch.matmul(concat_weather, self.W_weather_agg.weight.t()) + self.b_w

        return self.gnn_layer(
            u_emb, p_emb, t_emb, w_now_emb, edges,
            season_emb=s_emb, day_weather_emb=d_emb, month_weather_emb=m_emb,
            e_W_per_edge=e_W, use_season=self.use_season, is_baseline=False,
            skip_user_agg=self.skip_user_agg, use_time=self.use_time, use_weather=self.use_weather
        )

    def calc_translation_loss(self, head_idx, r_time_idx, r_weather_idx, tail_idx, neg_tail_idx, 
                              r_season_idx=None, r_day_seq_idx=None, r_month_idx=None,
                              u_emb_all=None, p_emb_all=None, t_emb_all=None, w_emb_all=None,
                              s_emb_all=None, d_emb_all=None, m_emb_all=None):
        
        if self.is_baseline:
            # Baseline: Time + Weather (Single)
            h = p_emb_all[head_idx] if p_emb_all is not None else self.poi_embedding(head_idx)
            t = p_emb_all[tail_idx] if p_emb_all is not None else self.poi_embedding(tail_idx)
            n_t = p_emb_all[neg_tail_idx] if p_emb_all is not None else self.poi_embedding(neg_tail_idx)
            r_time = t_emb_all[r_time_idx] if t_emb_all is not None else self.time_embedding(r_time_idx)
            r_weather = w_emb_all[r_weather_idx] if w_emb_all is not None else self.weather_now_embedding(r_weather_idx)
            if not self.use_time: r_time = 0
            if not self.use_weather: r_weather = 0
            relation_emb = r_time + r_weather
        else:
            # Proposed: Complex Relation (Time + Season + Multi-weather)
            h = p_emb_all[head_idx] if p_emb_all is not None else self.poi_embedding(head_idx)
            t = p_emb_all[tail_idx] if p_emb_all is not None else self.poi_embedding(tail_idx)
            n_t = p_emb_all[neg_tail_idx] if p_emb_all is not None else self.poi_embedding(neg_tail_idx)
            r_time = t_emb_all[r_time_idx] if t_emb_all is not None else self.time_embedding(r_time_idx)
            
            # Recalculate context embedding
            r_w_now = w_emb_all[r_weather_idx] if w_emb_all is not None else self.weather_now_embedding(r_weather_idx)
            if not self.use_time: r_time = 0
            if not self.use_weather: r_w_now = torch.zeros_like(r_w_now)
            
            if d_emb_all is not None:
                 r_w_minus = d_emb_all[r_day_seq_idx[:, 0]]; r_w_curr = d_emb_all[r_day_seq_idx[:, 1]]; r_w_plus = d_emb_all[r_day_seq_idx[:, 2]]
            else:
                 r_w_minus = self.day_weather_embedding(r_day_seq_idx[:, 0]); r_w_curr = self.day_weather_embedding(r_day_seq_idx[:, 1]); r_w_plus = self.day_weather_embedding(r_day_seq_idx[:, 2])
            
            concat_day = torch.cat([r_w_minus, r_w_curr, r_w_plus], dim=1)
            e_day = torch.matmul(concat_day, self.W_day_agg.weight.t()) + self.b_d
            
            e_month = m_emb_all[r_month_idx] if m_emb_all is not None else self.month_weather_embedding(r_month_idx)
            concat_weather = torch.cat([r_w_now, e_day, e_month], dim=1)
            e_W = torch.matmul(concat_weather, self.W_weather_agg.weight.t()) + self.b_w
            
            if self.use_season:
                 e_s = s_emb_all[r_season_idx] if s_emb_all is not None else self.season_embedding(r_season_idx)
                 relation_emb = r_time + e_s + e_W
            else:
                 relation_emb = r_time + e_W

        pos_dist = self.trans_layer(h, relation_emb, t)
        neg_dist = self.trans_layer(h, relation_emb, n_t)
        loss = torch.relu(pos_dist + self.gamma - neg_dist)
        return loss.mean()

    def enhance_poi_rep(self, traj_emb, mask=None):
        if mask is not None:
            mask_float = mask.unsqueeze(2).float() # [batch, seq, 1]
            masked_emb = traj_emb * mask_float
            avg_emb = masked_emb.sum(dim=1) / (mask_float.sum(dim=1) + 1e-9)
        else:
            avg_emb = traj_emb.mean(dim=1)
        return self.activation(self.W_poi_agg(avg_emb))

    def set_transition_neighbors(self, nbr_idx, nbr_mask):
        """POI遷移グラフ G_T の後続POI表を設定（graph_utils.build_transition_neighbors の出力）"""
        self.trans_nbr_idx, self.trans_nbr_mask = nbr_idx, nbr_mask

    def enhance_traj_poi(self, traj_poi_idx, e_r, p_all):
        """
        CAAI 4.1節 Eq(3) + 4.3節 Eq(8): 強化軌跡グラフ
          1. 軌跡上の各POI l_i について、G_T 上の遷移先候補 l から
             s(l_i, r_i, l) = exp(-||e_li + e_ri - e_l||^2) が大きい top-K を近傍 N(l_i) とする
             （r_i はその訪問時の時間・気象（・季節）文脈）
          2. e_li = σ(1/K Σ_{l∈N(l_i)} W1 e_l)
        遷移先を持たないPOI（パディング含む）は元の埋め込みをそのまま使う。
        """
        nbr_idx = self.trans_nbr_idx[traj_poi_idx]    # [B, L, D]
        nbr_mask = self.trans_nbr_mask[traj_poi_idx]  # [B, L, D]
        k = min(self.enhanced_traj_k, nbr_idx.size(-1))

        # 近傍選択（離散選択なので勾配は不要）: g が小さい = s が大きい
        with torch.no_grad():
            query = p_all[traj_poi_idx] + e_r                                  # [B, L, d]
            g = ((query.unsqueeze(2) - p_all[nbr_idx]) ** 2).sum(-1)           # [B, L, D]
            g = g.masked_fill(~nbr_mask, float('inf'))
            top = g.topk(k, dim=-1, largest=False).indices                     # [B, L, K]
            sel_idx = nbr_idx.gather(-1, top)
            sel_mask = nbr_mask.gather(-1, top).unsqueeze(-1).float()          # [B, L, K, 1]

        sel_emb = self.W_enh(p_all[sel_idx]) * sel_mask                        # [B, L, K, d]
        cnt = sel_mask.sum(2)                                                  # [B, L, 1]
        agg = self.activation(sel_emb.sum(2) / cnt.clamp(min=1))
        return torch.where(cnt > 0, agg, p_all[traj_poi_idx])

    def predict(self, user_idx, traj_poi_idx, traj_time_vec, traj_lat, traj_lon, candidate_poi_indices,
                cand_lat, cand_lon, traj_abs_time, target_abs_time, mask=None, p_emb_all=None, enhanced_history=None,
                traj_w_now=None, traj_w_day=None, traj_w_month=None, traj_season=None,
                u_emb_all=None, t_emb_all=None, w_emb_all=None,
                s_emb_all=None, d_emb_all=None, m_emb_all=None):
        """
        traj_time_vec: 時刻(0-24h)。タイムカテゴリ(24段階)の埋め込みにのみ使用
        traj_abs_time / target_abs_time: 絶対時間[h]。近接嗜好の d_t にのみ使用
        """
        # 1. Prepare Base Embeddings (GNN-refined if provided)
        u_emb = (u_emb_all[user_idx] if u_emb_all is not None else self.user_embedding(user_idx)).unsqueeze(1) # [batch, 1, dim]
        p_all = p_emb_all if p_emb_all is not None else self.poi_embedding.weight

        # 2. Build Trajectory Check-in Representation (Sum of all modalities)
        p_traj = p_all[traj_poi_idx]
        t_traj = t_emb_all[traj_time_vec.long()] if t_emb_all is not None else self.time_embedding(traj_time_vec.long())
        if not self.use_time: t_traj = 0

        if self.is_baseline:
            # Baseline: c_i = e_u + e_p + e_t + e_w   (e_r = e_t + e_w)
            if traj_w_now is None or not self.use_weather: w_traj = 0
            else: w_traj = w_emb_all[traj_w_now] if w_emb_all is not None else self.weather_now_embedding(traj_w_now)
            e_r = t_traj + w_traj
        else:
            # Proposed: c_i = e_u + e_p + e_t + e_s + e_W   (e_r = e_t + e_s + e_W)
            e_now = w_emb_all[traj_w_now] if w_emb_all is not None else self.weather_now_embedding(traj_w_now)
            if not self.use_weather: e_now = torch.zeros_like(e_now)
            
            # Day Weather
            if d_emb_all is not None:
                w_yest = d_emb_all[traj_w_day[:, :, 0]]; w_curr = d_emb_all[traj_w_day[:, :, 1]]; w_next = d_emb_all[traj_w_day[:, :, 2]]
            else:
                w_yest = self.day_weather_embedding(traj_w_day[:, :, 0])
                w_curr = self.day_weather_embedding(traj_w_day[:, :, 1])
                w_next = self.day_weather_embedding(traj_w_day[:, :, 2])
            concat_day = torch.cat([w_yest, w_curr, w_next], dim=-1)
            e_day = torch.matmul(concat_day, self.W_day_agg.weight.t()) + self.b_d
            
            e_month = m_emb_all[traj_w_month] if m_emb_all is not None else self.month_weather_embedding(traj_w_month)
            concat_weather = torch.cat([e_now, e_day, e_month], dim=-1)
            e_W = torch.matmul(concat_weather, self.W_weather_agg.weight.t()) + self.b_w
            
            if self.use_season:
                e_s = s_emb_all[traj_season] if s_emb_all is not None else self.season_embedding(traj_season)
                e_r = t_traj + e_s + e_W
            else:
                e_r = t_traj + e_W

        # 2.5 強化軌跡グラフ（CAAI Eq 3, 8）: 軌跡上のPOI表現を文脈付き遷移近傍の集約で置き換える
        if self.use_enhanced_traj:
            p_traj = self.enhance_traj_poi(traj_poi_idx, e_r, p_all)
        c_i = u_emb + p_traj + e_r

        # 3. SP-Attention to get History Representation (CAAI Eq 9-12, 近接嗜好は減算)
        history_rep = self.st_attention(c_i, traj_abs_time, traj_lat, traj_lon, mask=mask)
        if enhanced_history is not None and not self.skip_poi_agg:
            history_rep = history_rep + enhanced_history.unsqueeze(1)

        # 4. Build Candidate Representation (Q: POI Embedding Only as per paper)
        p_cand = p_all[candidate_poi_indices]

        # 5. 予測層: p_l = 1/|有効位置| Σ_j m_j W2_j (q_l·k_j/√d - w_s·log(1+d_s(l, l_j)))
        # history_rep(K): [batch, seq, dim], p_cand(Q): [batch, num_cands, dim] -> [batch, num_cands, seq]
        raw_scores = torch.matmul(p_cand, history_rep.transpose(1, 2)) / (self.embed_dim ** 0.5)

        # 空間的近接: 候補POIと各履歴POIの距離。
        # 時間間隔（正解時刻 - 履歴時刻）は候補POIによらず同じ値で、加算型では順位に影響しないため用いない
        # （target_abs_time は互換のため引数に残している）
        c_lat, c_lon = cand_lat.unsqueeze(2), cand_lon.unsqueeze(2) # [batch, num_cands, 1]
        h_lat, h_lon = traj_lat.unsqueeze(1), traj_lon.unsqueeze(1) # [batch, 1, seq]
        d_s = self.st_attention.tensor_haversine(c_lat, c_lon, h_lat, h_lon) # [batch, num_cands, seq]
        logits = raw_scores - self.st_attention.w_s * self.st_attention.spatial_feature(d_s)

        # 系列方向の集約: softmax なしの学習重み W2（STAN 公開実装と同様）。
        # 有効位置数で割るので、系列長によらずスコアのスケールが揃う
        m = mask.unsqueeze(1).float() if mask is not None else torch.ones_like(logits[:, :1, :])  # [batch, 1, seq]
        scores = (logits * self.W2 * m).sum(-1) / m.sum(-1)   # [batch, num_cands]
        return scores

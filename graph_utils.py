import torch

def build_interaction_graph(raw_data, device='cpu', session_ids=None):
    """
    Convert raw_data {user: [traj]} into edge tensors for GNN.
    Edge = (User, POI, Time, Weather)
    prev_poi_idxs / has_prev: 直前POI→現POI の遷移（POI遷移グラフ G_T の構築用）
    session_ids: {user: [軌跡ID, ...]}。指定時は軌跡の境界をまたぐ遷移を作らない
    """
    u_list, p_list, t_list, w_now_list, w_day_list, w_month_list, s_list = [], [], [], [], [], [], []
    prev_p_list, has_prev_list = [], []

    for u_id, traj in raw_data.items():
        sids = session_ids.get(u_id) if session_ids else None
        for i, item in enumerate(traj):
            # Supports both baseline (poi, time, weather, lat, lon, abs) and
            # proposed (poi, time, weather, lat, lon, yest, today, tom, month, season, abs) tuples
            poi, time_val, now_weather = item[0], item[1], item[2]

            # Map float time to index (0-23)
            time_idx = int(time_val) % 24

            if len(item) >= 11:
                three_day_weather = [item[5], item[6], item[7]]
                month_weather, season = item[8], item[9]
            else:
                three_day_weather, month_weather, season = [0, 0, 0], 0, 0

            u_list.append(u_id)
            p_list.append(poi)
            t_list.append(time_idx)
            w_now_list.append(now_weather)
            w_day_list.append(three_day_weather)
            w_month_list.append(month_weather)
            s_list.append(season)

            has_prev = i > 0 and (sids is None or sids[i-1] == sids[i])
            prev_p_list.append(traj[i-1][0] if has_prev else poi)
            has_prev_list.append(has_prev)

    edges = {
        'user_idxs': torch.tensor(u_list, dtype=torch.long).to(device),
        'poi_idxs': torch.tensor(p_list, dtype=torch.long).to(device),
        'prev_poi_idxs': torch.tensor(prev_p_list, dtype=torch.long).to(device),
        'has_prev': torch.tensor(has_prev_list, dtype=torch.bool).to(device),
        'time_idxs': torch.tensor(t_list, dtype=torch.long).to(device),
        'weather_now_idxs': torch.tensor(w_now_list, dtype=torch.long).to(device),
        'weather_idxs': torch.tensor(w_now_list, dtype=torch.long).to(device), # Alias for baseline
        'weather_day_idxs': torch.tensor(w_day_list, dtype=torch.long).to(device),
        'weather_month_idxs': torch.tensor(w_month_list, dtype=torch.long).to(device),
        'season_idxs': torch.tensor(s_list, dtype=torch.long).to(device),
    }
    return edges


def build_transition_neighbors(edges, num_pois):
    """
    POI遷移グラフ G_T から各POIの遷移先候補（後続POI）を作る（CAAI Fig.3 (b) の「遷移近傍」）。
    戻り値:
        nbr_idx  [num_pois, max_deg] long : POI l の後続POI（パディングは0）
        nbr_mask [num_pois, max_deg] bool : 有効な後続POIか
    """
    device = edges['poi_idxs'].device
    has_prev = edges['has_prev'].cpu()
    src = edges['prev_poi_idxs'].cpu()[has_prev]
    dst = edges['poi_idxs'].cpu()[has_prev]
    pairs = torch.unique(torch.stack([src, dst], dim=1), dim=0)  # src 昇順に並ぶ

    counts = torch.bincount(pairs[:, 0], minlength=num_pois)
    max_deg = max(int(counts.max()) if len(pairs) > 0 else 0, 1)
    offsets = torch.cumsum(counts, 0) - counts
    pos = torch.arange(len(pairs)) - offsets[pairs[:, 0]]

    nbr_idx = torch.zeros(num_pois, max_deg, dtype=torch.long)
    nbr_mask = torch.zeros(num_pois, max_deg, dtype=torch.bool)
    nbr_idx[pairs[:, 0], pos] = pairs[:, 1]
    nbr_mask[pairs[:, 0], pos] = True
    return nbr_idx.to(device), nbr_mask.to(device)

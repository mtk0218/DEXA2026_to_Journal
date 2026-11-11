import torch

def build_interaction_graph(raw_data, device='cpu', session_ids=None):
    """
    Convert raw_data {user: [traj]} into edge tensors for GNN.
    Edge = (User, POI, Time, Weather)
    session_ids: {user: [軌跡ID, ...]}。指定時は軌跡の境界をまたぐ遷移（prev_poi）を作らない
    """
    u_list, p_list, t_list, w_now_list, w_day_list, w_month_list, s_list = [], [], [], [], [], [], []
    prev_p_list = []
    prev_t_list = []
    prev_w_now_list = []
    prev_w_day_list = []
    prev_w_month_list = []
    prev_s_list = []
    
    for u_id, traj in raw_data.items():
        sids = session_ids.get(u_id) if session_ids else None
        for i, item in enumerate(traj):
            # Supports both 5-element (baseline) and 10-element (proposed) tuples
            poi, time_val, now_weather = item[0], item[1], item[2]
            
            # Map float time to index (0-23)
            time_idx = int(time_val) % 24 
            
            if len(item) >= 10:
                # Proposed: (poi, time, weather, lat, lon, yest, today, tom, month, season)
                three_day_weather = [item[5], item[6], item[7]]
                month_weather, season = item[8], item[9]
            else:
                # Baseline placeholders
                three_day_weather, month_weather, season = [0, 0, 0], 0, 0

            u_list.append(u_id)
            p_list.append(poi)
            t_list.append(time_idx)
            w_now_list.append(now_weather)
            w_day_list.append(three_day_weather)
            w_month_list.append(month_weather)
            s_list.append(season)
            
            # Track previous POI and previous contexts for enhanced trajectory graph filtering
            if i > 0 and (sids is None or sids[i-1] == sids[i]):
                prev_poi = traj[i-1][0]
                prev_time_val = traj[i-1][1]
                prev_time_idx = int(prev_time_val) % 24
                prev_w_now = traj[i-1][2]
                
                if len(traj[i-1]) >= 10:
                    prev_three_day = [traj[i-1][5], traj[i-1][6], traj[i-1][7]]
                    prev_month_w = traj[i-1][8]
                    prev_season = traj[i-1][9]
                else:
                    prev_three_day = [0, 0, 0]
                    prev_month_w = 0
                    prev_season = 0
            else:
                prev_poi = poi
                prev_time_idx = time_idx
                prev_w_now = now_weather
                prev_three_day = three_day_weather
                prev_month_w = month_weather
                prev_season = season
                
            prev_p_list.append(prev_poi)
            prev_t_list.append(prev_time_idx)
            prev_w_now_list.append(prev_w_now)
            prev_w_day_list.append(prev_three_day)
            prev_w_month_list.append(prev_month_w)
            prev_s_list.append(prev_season)
            
    edges = {
        'user_idxs': torch.tensor(u_list, dtype=torch.long).to(device),
        'poi_idxs': torch.tensor(p_list, dtype=torch.long).to(device),
        'prev_poi_idxs': torch.tensor(prev_p_list, dtype=torch.long).to(device),
        'time_idxs': torch.tensor(t_list, dtype=torch.long).to(device),
        'prev_time_idxs': torch.tensor(prev_t_list, dtype=torch.long).to(device),
        'weather_now_idxs': torch.tensor(w_now_list, dtype=torch.long).to(device),
        'prev_weather_now_idxs': torch.tensor(prev_w_now_list, dtype=torch.long).to(device),
        'weather_idxs': torch.tensor(w_now_list, dtype=torch.long).to(device), # Alias for baseline
        'weather_day_idxs': torch.tensor(w_day_list, dtype=torch.long).to(device),
        'prev_weather_day_idxs': torch.tensor(prev_w_day_list, dtype=torch.long).to(device),
        'weather_month_idxs': torch.tensor(w_month_list, dtype=torch.long).to(device),
        'prev_weather_month_idxs': torch.tensor(prev_w_month_list, dtype=torch.long).to(device),
        'season_idxs': torch.tensor(s_list, dtype=torch.long).to(device),
        'prev_season_idxs': torch.tensor(prev_s_list, dtype=torch.long).to(device),
    }
    return edges



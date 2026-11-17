import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from TWTransNet.twtransnet import TWTransNet
from TWTransNet.data_loader import CheckinDataset, get_eval_target_indices, to_abs_hours
from TWTransNet.utils import accuracy_at_k, mrr, ndcg_at_k
from graph_utils import build_interaction_graph, build_transition_neighbors
from datetime import datetime, timedelta
import sys
import os
import argparse
import random
import csv

# Global weather lists
WEATHER_LIST = ["Fair", "Partly Cloudy", "Cloudy", "Mostly Cloudy", "Light Rain", "Rain", "Heavy Rain", "Light Snow", "Snow", "Heavy Snow", "Fair / Windy", "Partly Cloudy / Windy", "Cloudy / Windy", "Mostly Cloudy / Windy", "Light Rain / Windy", "Rain / Windy", "Heavy Rain / Windy", "Light Snow / Windy", "Snow / Windy", "Heavy Snow / Windy"]
REALTIME_WEATHER_LIST = ["Fair", "Partly Cloudy", "Cloudy", "Mostly Cloudy", "Light Rain", "Rain", "Heavy Rain", "Light Snow", "Snow", "Heavy Snow", "Fair / Windy", "Partly Cloudy / Windy", "Cloudy / Windy", "Mostly Cloudy / Windy", "Light Rain / Windy", "Rain / Windy", "Heavy Rain / Windy", "Light Snow / Windy", "Snow / Windy", "Heavy Snow / Windy"]
DAY_WEATHER_LIST = ["Sunny", "Cloudy", "Rainy", "Snowy"]

def generate_mock_data(num_users=20, num_pois=100, min_len=5, max_len=20, existing=False):
    data, poi_coords = {}, {}
    for u in range(num_users):
        seq_len = np.random.randint(min_len, max_len)
        traj, abs_h = [], 0.0
        for _ in range(seq_len):
            abs_h += np.random.rand() * 48
            poi = np.random.randint(1, num_pois)
            time = np.random.rand() * 24
            now_weather_idx = np.random.randint(0, len(REALTIME_WEATHER_LIST))
            lat, lon = np.random.uniform(-90, 90), np.random.uniform(-180, 180)
            if existing:
                traj.append((poi, time, now_weather_idx, lat, lon, abs_h))
            else:
                yest, today, tom = np.random.randint(0, 4), np.random.randint(0, 4), np.random.randint(0, 4)
                month_w, season = np.random.randint(0, 9), np.random.randint(0, 4)
                traj.append((poi, time, now_weather_idx, lat, lon, yest, today, tom, month_w, season, abs_h))
            poi_coords[poi] = (float(lat), float(lon))
        data[u] = traj
    return len(data), num_pois, data, poi_coords

def next_session_id(sids, last_dt, dt, session_gap_hours):
    """直前の訪問から session_gap_hours を超えて空いていれば新しい軌跡IDを振る（None/0以下なら分割しない）"""
    if not sids: return 0
    if session_gap_hours and session_gap_hours > 0 and (dt - last_dt).total_seconds() > session_gap_hours * 3600:
        return sids[-1] + 1
    return sids[-1]

def load_real_data(city, existing=False, session_gap_hours=0): 
    data, POI_list, user_map, traj, current_user_idx, last_raw_user_id, poi_coords = {}, [], {}, [], -1, None, {}
    # session_ids[u][i]: 同一ユーザ内の軌跡ID（直前の訪問から session_gap_hours 超空いたら新しい軌跡）
    session_ids, sids, last_dt = {}, [], None

    if city == "Boston_all":
        file_path = "boston_1year_checkin_weather.csv"
    else:
        file_path = "Boston_tourism_weather.csv"

    with open(file_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split(",")
            if existing:
                if len(parts) < 9: continue
                user_id, poi, UTC_time, Timezone, lat, lon, _, _, weather = parts[:9]
            else:
                if len(parts) < 14: continue
                user_id, poi, UTC_time, Timezone, lat, lon, _, _, weather, yesterday, today, tomorrow, month_w, season = parts[:14]
            if user_id not in user_map: user_map[user_id] = len(user_map)
            u_idx = user_map[user_id]
            if last_raw_user_id is None: last_raw_user_id, current_user_idx = user_id, u_idx
            elif last_raw_user_id != user_id:
                data[current_user_idx], session_ids[current_user_idx] = traj, sids
                traj, sids, last_dt, last_raw_user_id, current_user_idx = [], [], None, user_id, u_idx
            if poi not in POI_list: POI_list.append(poi)
            POI_index = POI_list.index(poi) + 1
            poi_coords[POI_index] = (float(lat), float(lon))
            utc_dt = datetime.strptime(UTC_time, "%a %b %d %H:%M:%S %z %Y")
            sids.append(next_session_id(sids, last_dt, utc_dt, session_gap_hours)); last_dt = utc_dt
            try: now_weather_index = REALTIME_WEATHER_LIST.index(weather) if not existing else WEATHER_LIST.index(weather)
            except ValueError: now_weather_index = 0
            local_time = utc_dt + timedelta(minutes=int(Timezone))
            float_time = local_time.hour + local_time.minute / 60
            if existing: traj.append((POI_index, float_time, now_weather_index, float(lat), float(lon), to_abs_hours(utc_dt)))
            else:
                try: yesterday_idx = DAY_WEATHER_LIST.index(yesterday); today_idx = DAY_WEATHER_LIST.index(today); tomorrow_idx = DAY_WEATHER_LIST.index(tomorrow)
                except: yesterday_idx, today_idx, tomorrow_idx = 0, 0, 0
                traj.append((POI_index, float_time, now_weather_index, float(lat), float(lon), yesterday_idx, today_idx, tomorrow_idx, int(month_w), int(season), to_abs_hours(utc_dt)))
    if current_user_idx != -1: data[current_user_idx], session_ids[current_user_idx] = traj, sids
    return len(data), len(POI_list) + 1, data, poi_coords, session_ids

def load_NYC_data(existing=False, session_gap_hours=0):
    data, poi_map, user_map, poi_coords = {}, {}, {}, {}
    session_ids, last_datetimes = {}, {}
    weather_map = {"Clear": 0, "Cloudy": 1, "Rain": 2, "Snow": 3}
    with open("NYC_checkin.csv", "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {
            "user_id", "venue_id", "category_id", "category", "lat", "lon",
            "local_time", "now_condition", "yesterday_condition",
            "today_condition", "tomorrow_condition", "monthly_condition", "season",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"NYC_checkin.csv is missing columns: {', '.join(sorted(missing_columns))}")

        for line_num, row in enumerate(reader, start=2):
            user_id, venue_id = row["user_id"], row["venue_id"]
            user_idx = user_map.setdefault(user_id, len(user_map))
            poi_idx = poi_map.setdefault(venue_id, len(poi_map) + 1)
            data.setdefault(user_idx, [])
            session_ids.setdefault(user_idx, [])

            try:
                lat, lon = float(row["lat"]), float(row["lon"])
                visit_time = None
                for time_format in ("%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
                    try:
                        visit_time = datetime.strptime(row["local_time"], time_format)
                        break
                    except ValueError:
                        continue
                if visit_time is None:
                    raise ValueError(f"unsupported local_time {row['local_time']!r}")

                now_weather = weather_map[row["now_condition"]]
                if existing:
                    trajectory_item = (
                        poi_idx,
                        visit_time.hour + visit_time.minute / 60,
                        now_weather,
                        lat,
                        lon,
                        to_abs_hours(visit_time),
                    )
                else:
                    yesterday = weather_map[row["yesterday_condition"]]
                    today = weather_map[row["today_condition"]]
                    tomorrow = weather_map[row["tomorrow_condition"]]
                    monthly_condition = int(row["monthly_condition"])
                    season = int(row["season"])
                    if not 0 <= season < 4:
                        raise ValueError(f"season must be between 0 and 3, got {season}")
                    if monthly_condition < 0:
                        raise ValueError(f"monthly_condition must be non-negative, got {monthly_condition}")
                    trajectory_item = (
                        poi_idx,
                        visit_time.hour + visit_time.minute / 60,
                        now_weather,
                        lat,
                        lon,
                        yesterday,
                        today,
                        tomorrow,
                        monthly_condition,
                        season,
                        to_abs_hours(visit_time),
                    )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid NYC_checkin.csv row {line_num}: {exc}") from exc

            poi_coords[poi_idx] = (lat, lon)
            previous_visits = session_ids[user_idx]
            last_dt = last_datetimes.get(user_idx)
            previous_visits.append(next_session_id(previous_visits, last_dt, visit_time, session_gap_hours))
            last_datetimes[user_idx] = visit_time
            data[user_idx].append(trajectory_item)

    return len(user_map), len(poi_map) + 1, data, poi_coords, session_ids

def load_Kyoto_data(existing=False, return_poi_info=False, session_gap_hours=0):
    data, POI_list, user_map, traj, current_user_idx, last_raw_user_id, poi_coords = {}, [], {}, [], -1, None, {}
    session_ids, sids, last_dt = {}, [], None
    poi_names = {}  # POI_index -> matched_aoi (実際の観光地名)
    import csv
    if not os.path.exists("Kyoto_trip_aoi_min4.csv"):
        print("Error: Kyoto data file not found.")
        return 0, 0, {}, {}
    with open("Kyoto_trip_aoi_min4.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for parts in reader:
            if not parts or parts[0] == "uid": continue
            if existing:
                if len(parts) < 7: continue
                user_id, lat, lon, aoi_id, matched_aoi, time, now_w = parts[:7]
                poi = aoi_id
            else:
                if len(parts) < 12: continue
                user_id, lat, lon, aoi_id, matched_aoi, time, now_w, yesterday, today, tom, month_w, season = parts[:12]
                poi = aoi_id
            if user_id not in user_map: user_map[user_id] = len(user_map)
            u_idx = user_map[user_id]
            if last_raw_user_id is None: last_raw_user_id, current_user_idx = user_id, u_idx
            elif last_raw_user_id != user_id:
                data[current_user_idx], session_ids[current_user_idx] = traj, sids
                traj, sids, last_dt, last_raw_user_id, current_user_idx = [], [], None, user_id, u_idx
            if poi not in POI_list: POI_list.append(poi)
            POI_index = POI_list.index(poi) + 1
            poi_coords[POI_index] = (float(lat), float(lon))
            poi_names[POI_index] = matched_aoi
            now_weather_index = int(now_w) - 1
            local_time = datetime.strptime(time, "%Y/%m/%d %H:%M")
            sids.append(next_session_id(sids, last_dt, local_time, session_gap_hours)); last_dt = local_time
            float_time = local_time.hour + local_time.minute / 60
            if existing: traj.append((POI_index, float_time, now_weather_index, float(lat), float(lon), to_abs_hours(local_time)))
            else: traj.append((POI_index, float_time, now_weather_index, float(lat), float(lon), int(yesterday)-11, int(today)-11, int(tom)-11, int(month_w)-21, int(season), to_abs_hours(local_time)))
    if current_user_idx != -1: data[current_user_idx], session_ids[current_user_idx] = traj, sids
    if return_poi_info:
        return len(data), len(POI_list) + 1, data, poi_coords, POI_list, poi_names, session_ids
    return len(data), len(POI_list) + 1, data, poi_coords, session_ids

def compute_dist_scale(dataset, seq_len, pct=95):
    """学習サンプルの入力系列内ペア (i<j) の距離[km]・時間差[h] の pct パーセンタイルを返す（正規化の上限 D_max）"""
    ds_all, dt_all = [], []
    for smp in dataset.data_samples:
        seq = smp['input_seq'][-seq_len:]
        if len(seq) < 2: continue
        lat = np.radians([x[3] for x in seq]); lon = np.radians([x[4] for x in seq]); t = np.array([x[-1] for x in seq], dtype=np.float64)
        i, j = np.triu_indices(len(seq), k=1)
        a = np.sin((lat[j]-lat[i])/2)**2 + np.cos(lat[i])*np.cos(lat[j])*np.sin((lon[j]-lon[i])/2)**2
        ds_all.append(2*6371.0*np.arcsin(np.sqrt(np.clip(a, 0, 1)))); dt_all.append(np.abs(t[j]-t[i]))
    ds_all, dt_all = np.concatenate(ds_all), np.concatenate(dt_all)
    return max(float(np.percentile(ds_all, pct)), 1e-3), max(float(np.percentile(dt_all, pct)), 1e-3)

class TransNegSampler:
    """翻訳損失（CAAI Eq. 2）の負例 tail をサンプリングする。CAAI Sec. 4.1 の制約:
      (a) l_n != l_j（正解 tail と異なる）
      (b) l_n はそのユーザが同じ時間・天気の文脈 r_i で遷移したことのある POI ではない
    (b) の既知遷移は学習データ（train split）の (user, r_i(head), target) からのみ作る（val/test の正解は使わない）。
    文脈 r_i はモデルの関係ベクトルに実際に入る要素で定義する（--no_time / --no_weather / --no_season で外した要素は除く）。"""
    def __init__(self, train_dataset, num_pois, model, max_tries=50, device='cpu'):
        self.num_pois, self.max_tries, self.device = num_pois, max_tries, device
        self.is_baseline = model.is_baseline
        self.use_time, self.use_weather, self.use_season = model.use_time, model.use_weather, model.use_season
        self.key2id, codes = {}, []
        for smp in train_dataset.data_samples:
            h, tgt = smp['input_seq'][-1], smp['target']
            has_full = len(h) >= 11
            key = self._key(smp['user_id'], int(h[1]), h[2],
                            *(h[5:10] if has_full else (0, 0, 0, 0, 0)))
            cid = self.key2id.setdefault(key, len(self.key2id))
            codes.append(cid * num_pois + int(tgt[0]))
        self.forbidden = torch.unique(torch.tensor(codes, dtype=torch.long)).to(device)
        print(f"TransNegSampler: contexts(user, r)={len(self.key2id)}, known transitions={self.forbidden.numel()}")

    def _key(self, user, time_idx, now_w, y, t, tm, month, season):
        k = [int(user), int(time_idx) if self.use_time else -1, int(now_w) if self.use_weather else -1]
        if not self.is_baseline:
            k += [int(y), int(t), int(tm), int(month), int(season) if self.use_season else -1]
        return tuple(k)

    def sample(self, batch, target_poi):
        """batch の各サンプルについて負例 tail を1つ返す（POI index 1..num_pois-1）"""
        n, device = target_poi.size(0), target_poi.device
        cols = [batch['user_id'], batch['head_time'].long(), batch['head_now_weather'],
                batch['head_yesterday_weather'], batch['head_today_weather'], batch['head_tomorrow_weather'],
                batch['head_month_weather'], batch['head_season']]
        rows = torch.stack([c.cpu().long() for c in cols], dim=1).tolist()
        cid = torch.tensor([self.key2id.get(self._key(*r), -1) for r in rows], dtype=torch.long, device=device)
        forbidden = self.forbidden.to(device)
        neg = torch.randint(1, self.num_pois, (n,), device=device)
        for _ in range(self.max_tries):
            bad = (neg == target_poi) | ((cid >= 0) & torch.isin(cid * self.num_pois + neg, forbidden))
            if not bad.any(): break
            neg[bad] = torch.randint(1, self.num_pois, (int(bad.sum()),), device=device)
        # (a) だけは必ず満たす（(b) を満たす候補が見つからない極端な場合のフォールバック）
        same = neg == target_poi
        if same.any(): neg[same] = target_poi[same] % (self.num_pois - 1) + 1
        return neg

def draw_trans_neg(trans_sampler, batch, target_poi, num_pois):
    if trans_sampler is None:  # 従来: 制約なしの一様サンプリング
        return torch.randint(1, num_pois, target_poi.shape, device=target_poi.device)
    return trans_sampler.sample(batch, target_poi)

REC_NUM_NEG = 10  # CAAI 論文 Sec. 4: "The number of negative samples is set to 10."

def sample_negatives(target_idx, num_cands, num_neg=REC_NUM_NEG):
    """正解以外の候補から重複なしで一様に num_neg 個の負例インデックスを引く（ステップごとに引き直す）。
    一様乱数の上位 num_neg 個 = 重複なし一様サンプリング。正解は -1 にして選ばれないようにする"""
    num_neg = min(num_neg, num_cands - 1)
    with torch.no_grad():
        r = torch.rand(target_idx.size(0), num_cands, device=target_idx.device)
        r.scatter_(1, target_idx.unsqueeze(1), -1.0)
        return r.topk(num_neg, dim=1).indices  # [B, num_neg]

def rec_loss_sampled_ce(scores, target_idx, num_neg=REC_NUM_NEG):
    """STAN 公開実装（train.py の sampling_prob + F.cross_entropy）と同じ損失:
    [正解, 負例 num_neg 個] のスコアに softmax をかけた CrossEntropy（正解は 0 列目）。
    公開実装は batch_size=1 なので、サンプルごとに 1+num_neg 列で計算するのと等価"""
    pos = scores.gather(1, target_idx.unsqueeze(1))                                   # [B, 1]
    neg = scores.gather(1, sample_negatives(target_idx, scores.size(1), num_neg))     # [B, num_neg]
    logits = torch.cat([pos, neg], dim=1)                                             # [B, 1+num_neg]
    return F.cross_entropy(logits, torch.zeros_like(target_idx))

def rec_loss_neg_sampling(scores, target_idx, num_neg=REC_NUM_NEG):
    """Balanced sampler（STAN Eq 11 / CAAI Eq 14）: L = -Σ( log σ(a_k) + Σ_{(j1..js)≠k} log(1 - σ(a_j)) )
    scores: [B, C] 全候補のスコア, target_idx: [B] 正解の候補インデックス
    負例は正解以外の候補から重複なしで一様に num_neg 個サンプリングする（ステップごとに引き直す。バッチ平均を返す）"""
    pos = scores.gather(1, target_idx.unsqueeze(1)).squeeze(1)  # [B]
    neg = scores.gather(1, sample_negatives(target_idx, scores.size(1), num_neg))  # [B, num_neg]
    # log(1 - σ(x)) = log σ(-x)
    loss = -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(dim=1))
    return loss.mean()

def rec_loss_full(scores, target_idx):
    """サンプラーなしの損失（STAN Eq 10）: 正解以外の全候補を負例とする。
    STAN 5.3節 "For evaluation, we drop the balanced sampler" に従い、評価時はこちらを使う（決定的に計算される）"""
    pos = scores.gather(1, target_idx.unsqueeze(1)).squeeze(1)  # [B]
    neg_mask = torch.ones_like(scores, dtype=torch.bool).scatter_(1, target_idx.unsqueeze(1), False)
    neg_term = (F.logsigmoid(-scores) * neg_mask).sum(dim=1)
    return (-(F.logsigmoid(pos) + neg_term)).mean()

def compute_rec_loss(scores, target_idx, rec_loss_type='sampled_ce', training=True):
    """推薦損失の切り替え（評価時はいずれもサンプラーを外す。STAN 5.3節）:
      'sampled_ce' = STAN 公開実装: [正解+負例10個] の softmax CrossEntropy。評価時は全候補の softmax CE
      'neg'        = 論文の式（STAN Eq 11 / CAAI Eq 14）の σ 形式。評価時は STAN Eq 10
      'ce'         = 全候補 softmax の CrossEntropy（旧版）
    Acc / MRR / nDCG はどの場合も全候補のスコアで計算する"""
    if rec_loss_type == 'ce' or (rec_loss_type == 'sampled_ce' and not training):
        return F.cross_entropy(scores, target_idx)
    if rec_loss_type == 'sampled_ce':
        return rec_loss_sampled_ce(scores, target_idx)
    return rec_loss_neg_sampling(scores, target_idx) if training else rec_loss_full(scores, target_idx)

class AutoLossWeight(nn.Module):
    """λ の自動決定: 不確実性に基づくマルチタスク重み付け (Kendall et al., CVPR 2018)
    L = exp(-s_rec)·L_rec + s_rec + exp(-s_trans)·L_trans + s_trans
    s_i = log σ_i^2 は学習可能パラメータ。損失が大きい（不確実な）タスクほど重みが自動的に小さくなる。
    実効的な λ（L_rec に対する L_trans の相対重み）は exp(s_rec - s_trans)。
    L_trans はヒンジ損失で 0 になりうるため（このとき s_trans → -∞ で重みが発散する）、s を [s_min, s_max] にクランプする。"""
    def __init__(self, init_lambda=1.0, s_min=-4.0, s_max=4.0, max_lambda=None):
        super().__init__()
        # max_lambda: 実効 λ の上限（L_trans が学習データ上で 0 に張り付き λ が際限なく増えるのを防ぐ）
        self.log_max_lambda = None if max_lambda is None else float(np.log(max_lambda))
        # 初期値: s_rec=0, s_trans=-log(init_lambda) とし、学習開始時の実効 λ を init_lambda に合わせる
        self.log_vars = nn.Parameter(torch.tensor([0.0, -float(np.log(init_lambda))]))
        self.s_min, self.s_max = s_min, s_max

    def _s(self):
        s = self.log_vars.clamp(self.s_min, self.s_max)
        if self.log_max_lambda is not None:
            # λ = exp(s_rec - s_trans) <= max_lambda  ⇔  s_trans >= s_rec - log(max_lambda)
            s = torch.stack([s[0], torch.maximum(s[1], s[0] - self.log_max_lambda)])
        return s

    def forward(self, rec_loss, trans_loss):
        s = self._s()
        return torch.exp(-s[0]) * rec_loss + s[0] + torch.exp(-s[1]) * trans_loss + s[1]

    def effective_lambda(self):
        s = self._s().detach()
        return float(torch.exp(s[0] - s[1]))

def train_epoch(model, dataloader, optimizer, num_pois, edges, poi_coords, lambda_weight=0.1, device='cpu', skip_poi_agg=False, skip_trans_loss=False, loss_weighter=None, rec_loss_type='sampled_ce', trans_sampler=None):
    model.train()
    total_loss, total_rec_loss, total_trans_loss = 0, 0, 0
    total_acc1, total_acc5, total_acc10, total_mrr, total_ndcg10, total_acc20 = 0, 0, 0, 0, 0, 0
    total_samples = 0
    coord_tensor = torch.zeros(num_pois, 2, device=device)
    for idx, (lat, lon) in poi_coords.items():
        if idx < num_pois: coord_tensor[idx] = torch.tensor([lat, lon], device=device)
    
    for batch in dataloader:
        user_id, traj_poi, traj_time, traj_lat, traj_lon = batch['user_id'].to(device), batch['traj_poi'].to(device), batch['traj_time'].to(device), batch['traj_lat'].to(device), batch['traj_lon'].to(device)
        target_poi, target_time = batch['target_poi'].to(device), batch['target_time'].to(device)
        head_poi, head_time, head_now_weather = batch['head_poi'].to(device), batch['head_time'].to(device), batch['head_now_weather'].to(device)
        mask = batch['mask'].to(device)
        optimizer.zero_grad()
        
        gnn_out = model.forward_gnn(edges)
        if len(gnn_out) == 4: (refined_u, refined_p, refined_t, refined_w), refined_s, refined_day, refined_month = gnn_out, None, None, None
        else: refined_u, refined_p, refined_t, refined_w, refined_s, refined_day, refined_month = gnn_out
        
        cands = torch.arange(1, num_pois).unsqueeze(0).repeat(user_id.size(0), 1).to(device)
        cand_coords = coord_tensor[cands]
        cand_lats, cand_lons = cand_coords[:, :, 0], cand_coords[:, :, 1]
        
        if skip_poi_agg:
            enhanced_history = None
        else:
            traj_emb_for_agg = refined_p[traj_poi]
            enhanced_history = model.enhance_poi_rep(traj_emb_for_agg, mask=mask)
        
        traj_w_day = torch.stack([batch['traj_yesterday_weather'].to(device), batch['traj_today_weather'].to(device), batch['traj_tomorrow_weather'].to(device)], dim=2)
        target_w_day = torch.stack([batch['target_yesterday_weather'].to(device), batch['target_today_weather'].to(device), batch['target_tomorrow_weather'].to(device)], dim=1)
        
        scores = model.predict(
            user_id, traj_poi, traj_time, traj_lat, traj_lon, cands, cand_lats, cand_lons,
            batch['traj_abs_time'].to(device), batch['target_abs_time'].to(device),
            mask=mask, p_emb_all=refined_p, enhanced_history=enhanced_history,
            u_emb_all=refined_u, t_emb_all=refined_t, w_emb_all=refined_w, s_emb_all=refined_s, d_emb_all=refined_day, m_emb_all=refined_month,
            traj_w_now=batch['traj_now_weather'].to(device), traj_w_day=traj_w_day, traj_w_month=batch['traj_month_weather'].to(device), traj_season=batch['traj_season'].to(device)
        )
        target_idx = target_poi - 1
        rec_loss = compute_rec_loss(scores, target_idx, rec_loss_type)

        if skip_trans_loss:
            trans_loss = torch.tensor(0.0, device=device)
        else:
            neg_tail = draw_trans_neg(trans_sampler, batch, target_poi, num_pois)
            if model.is_baseline:
                trans_loss = model.calc_translation_loss(head_poi, head_time.long(), head_now_weather, target_poi, neg_tail, u_emb_all=refined_u, p_emb_all=refined_p, t_emb_all=refined_t, w_emb_all=refined_w)
            else:
                head_w_day_seq = torch.stack([batch['head_yesterday_weather'].to(device), batch['head_today_weather'].to(device), batch['head_tomorrow_weather'].to(device)], dim=1)
                trans_loss = model.calc_translation_loss(head_poi, head_time.long(), head_now_weather, target_poi, neg_tail, r_season_idx=batch['head_season'].to(device), r_day_seq_idx=head_w_day_seq, r_month_idx=batch['head_month_weather'].to(device), u_emb_all=refined_u, p_emb_all=refined_p, t_emb_all=refined_t, w_emb_all=refined_w, s_emb_all=refined_s, d_emb_all=refined_day, m_emb_all=refined_month)
        
        if loss_weighter is not None and not skip_trans_loss: loss = loss_weighter(rec_loss, trans_loss)
        else: loss = rec_loss + lambda_weight * trans_loss
        loss.backward(); optimizer.step()
        if loss_weighter is not None and not skip_trans_loss: loss = rec_loss.detach() + loss_weighter.effective_lambda() * trans_loss.detach()
        
        bs = user_id.size(0)
        total_samples += bs
        total_loss += loss.item() * bs; total_rec_loss += rec_loss.item() * bs; total_trans_loss += trans_loss.item() * bs
        total_acc1 += accuracy_at_k(scores, target_idx, k=1) * bs; total_acc5 += accuracy_at_k(scores, target_idx, k=5) * bs; total_acc10 += accuracy_at_k(scores, target_idx, k=10) * bs
        total_mrr += mrr(scores, target_idx) * bs; total_ndcg10 += ndcg_at_k(scores, target_idx, k=10) * bs
        total_acc20 += accuracy_at_k(scores, target_idx, k=min(20, scores.size(1))) * bs
        
    return (total_loss/total_samples, total_rec_loss/total_samples, total_trans_loss/total_samples, total_acc1/total_samples, total_acc5/total_samples, total_acc10/total_samples, total_mrr/total_samples, total_ndcg10/total_samples, total_acc20/total_samples) if total_samples > 0 else (0,0,0,0,0,0,0,0,0)

def evaluate(model, dataloader, num_pois, edges, poi_coords, lambda_weight=0.1, device='cpu', skip_poi_agg=False, skip_trans_loss=False, loss_weighter=None, rec_loss_type='sampled_ce', trans_sampler=None):
    model.eval()
    total_loss, total_rec_loss, total_trans_loss = 0, 0, 0
    total_acc1, total_acc5, total_acc10, total_mrr, total_ndcg10, total_acc20 = 0, 0, 0, 0, 0, 0
    total_samples = 0
    coord_tensor = torch.zeros(num_pois, 2, device=device)
    for idx, (lat, lon) in poi_coords.items():
        if idx < num_pois: coord_tensor[idx] = torch.tensor([lat, lon], device=device)
    
    with torch.no_grad():
        for batch in dataloader:
            user_id, traj_poi, traj_time, traj_lat, traj_lon = batch['user_id'].to(device), batch['traj_poi'].to(device), batch['traj_time'].to(device), batch['traj_lat'].to(device), batch['traj_lon'].to(device)
            target_poi, target_time = batch['target_poi'].to(device), batch['target_time'].to(device)
            head_poi, head_time, head_now_weather = batch['head_poi'].to(device), batch['head_time'].to(device), batch['head_now_weather'].to(device)
            mask = batch['mask'].to(device)
            
            gnn_out = model.forward_gnn(edges)
            if len(gnn_out) == 4: (refined_u, refined_p, refined_t, refined_w), refined_s, refined_day, refined_month = gnn_out, None, None, None
            else: refined_u, refined_p, refined_t, refined_w, refined_s, refined_day, refined_month = gnn_out
            
            cands = torch.arange(1, num_pois).unsqueeze(0).repeat(user_id.size(0), 1).to(device)
            cand_coords = coord_tensor[cands]
            cand_lats, cand_lons = cand_coords[:, :, 0], cand_coords[:, :, 1]
            
            if skip_poi_agg:
                enhanced_history = None
            else:
                traj_emb_for_agg = refined_p[traj_poi]
                enhanced_history = model.enhance_poi_rep(traj_emb_for_agg, mask=mask)
            traj_w_day = torch.stack([batch['traj_yesterday_weather'].to(device), batch['traj_today_weather'].to(device), batch['traj_tomorrow_weather'].to(device)], dim=2)
            target_w_day = torch.stack([batch['target_yesterday_weather'].to(device), batch['target_today_weather'].to(device), batch['target_tomorrow_weather'].to(device)], dim=1)
            
            scores = model.predict(
                user_id, traj_poi, traj_time, traj_lat, traj_lon, cands, cand_lats, cand_lons,
            batch['traj_abs_time'].to(device), batch['target_abs_time'].to(device),
                mask=mask, p_emb_all=refined_p, enhanced_history=enhanced_history,
                u_emb_all=refined_u, t_emb_all=refined_t, w_emb_all=refined_w, s_emb_all=refined_s, d_emb_all=refined_day, m_emb_all=refined_month,
                traj_w_now=batch['traj_now_weather'].to(device), traj_w_day=traj_w_day, traj_w_month=batch['traj_month_weather'].to(device), traj_season=batch['traj_season'].to(device)
            )
            target_idx = target_poi - 1
            rec_loss = compute_rec_loss(scores, target_idx, rec_loss_type, training=False)
            
            if skip_trans_loss:
                trans_loss = torch.tensor(0.0, device=device)
            else:
                neg_tail = draw_trans_neg(trans_sampler, batch, target_poi, num_pois)
                if model.is_baseline:
                    trans_loss = model.calc_translation_loss(head_poi, head_time.long(), head_now_weather, target_poi, neg_tail, u_emb_all=refined_u, p_emb_all=refined_p, t_emb_all=refined_t, w_emb_all=refined_w)
                else:
                    head_w_day_seq = torch.stack([batch['head_yesterday_weather'].to(device), batch['head_today_weather'].to(device), batch['head_tomorrow_weather'].to(device)], dim=1)
                    trans_loss = model.calc_translation_loss(head_poi, head_time.long(), head_now_weather, target_poi, neg_tail, r_season_idx=batch['head_season'].to(device), r_day_seq_idx=head_w_day_seq, r_month_idx=batch['head_month_weather'].to(device), u_emb_all=refined_u, p_emb_all=refined_p, t_emb_all=refined_t, w_emb_all=refined_w, s_emb_all=refined_s, d_emb_all=refined_day, m_emb_all=refined_month)
            
            if loss_weighter is not None and not skip_trans_loss: loss = rec_loss + loss_weighter.effective_lambda() * trans_loss
            else: loss = rec_loss + lambda_weight * trans_loss
            bs = user_id.size(0)
            total_samples += bs
            total_loss += loss.item() * bs; total_rec_loss += rec_loss.item() * bs; total_trans_loss += trans_loss.item() * bs
            total_acc1 += accuracy_at_k(scores, target_idx, k=1) * bs; total_acc5 += accuracy_at_k(scores, target_idx, k=5) * bs; total_acc10 += accuracy_at_k(scores, target_idx, k=10) * bs
            total_mrr += mrr(scores, target_idx) * bs; total_ndcg10 += ndcg_at_k(scores, target_idx, k=10) * bs
            total_acc20 += accuracy_at_k(scores, target_idx, k=min(20, scores.size(1))) * bs
            
    return (total_loss/total_samples, total_rec_loss/total_samples, total_trans_loss/total_samples, total_acc1/total_samples, total_acc5/total_samples, total_acc10/total_samples, total_mrr/total_samples, total_ndcg10/total_samples, total_acc20/total_samples) if total_samples > 0 else (0,0,0,0,0,0,0,0,0)

def main(lamda_weight, city='A', resume=True, seed=42, use_season=True, use_gnn=True, existing=False, 
         skip_user_agg=False, skip_poi_agg=False, skip_trans_loss=False, use_enhanced_traj=False, enhanced_traj_k=6, epochs=100, session_gap_hours=0, split_scope='both', leak=False,
         use_time=True, use_weather=True, dist_norm='legacy', dist_pct=95, auto_lambda=False, auto_lambda_max=None, rec_loss='sampled_ce', seq_len=10, select_metric='ndcg10', trans_neg='constrained'):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    POI_list, poi_names = None, None  # ケーススタディ用: POI index -> aoi_id / 観光地名
    Config = {
        'BATCH_SIZE': 16, 'EMBED_DIM': 32, 'LR': 0.001, 'EPOCHS': epochs, 'SEQ_LEN': seq_len, 'DROPOUT': 0.1, 
        'GAMMA': 1.0, 'LAMBDA_WEIGHT': lamda_weight, 'CITY': city, 'USE_SEASON': use_season, 'USE_GNN': use_gnn, 
        'USE_TIME': use_time, 'USE_WEATHER': use_weather,
        'SKIP_USER_AGG': skip_user_agg, 'SKIP_POI_AGG': skip_poi_agg, 'SKIP_TRANS_LOSS': skip_trans_loss,
        'USE_ENHANCED_TRAJ': use_enhanced_traj, 'ENHANCED_TRAJ_K': enhanced_traj_k,
        'SESSION_GAP_HOURS': session_gap_hours, 'SPLIT_SCOPE': split_scope, 'LEAK': leak,
        'DIST_NORM': dist_norm, 'DIST_PCT': dist_pct, 'REC_LOSS': rec_loss,
        'SELECT_METRIC': select_metric
    }
    session_ids = None  # None の場合はユーザ単位の1本の軌跡として扱う
    
    if city == "NYC":
        num_users, num_pois, raw_data, poi_coords, session_ids = load_NYC_data(existing=existing, session_gap_hours=session_gap_hours)
        num_now_weathers = 4
        num_day_weathers, num_month_weathers = (1, 1) if existing else (4, 5)
    elif city == "Kyoto":
        num_users, num_pois, raw_data, poi_coords, POI_list, poi_names, session_ids = load_Kyoto_data(existing=existing, return_poi_info=True, session_gap_hours=session_gap_hours)
        num_now_weathers, num_day_weathers, num_month_weathers = 10, (1 if existing else 4), (1 if existing else 4)
    elif city == "Boston_all" or city == "Boston_tour":
        num_users, num_pois, raw_data, poi_coords, session_ids = load_real_data(city, existing=existing, session_gap_hours=session_gap_hours)
        num_now_weathers = len(REALTIME_WEATHER_LIST) if not existing else len(WEATHER_LIST)
        num_day_weathers, num_month_weathers = (1 if existing else len(DAY_WEATHER_LIST)), (1 if existing else 9)
    else:
        num_users, num_pois, raw_data, poi_coords = generate_mock_data(existing=existing)
        num_now_weathers, num_day_weathers, num_month_weathers = len(REALTIME_WEATHER_LIST), (1 if existing else 4), (1 if existing else 9)
    
    print(f"Loaded {city} Data: Users={num_users}, POIs={num_pois}")
    num_sessions = sum(len(set(s)) for s in session_ids.values()) if session_ids else num_users
    split_by_gap = num_sessions > num_users  # 実際に軌跡が分割されたか
    # split_scope: 軌跡の区切りを入力系列(input)・グラフの遷移エッジ(graph)のどちらに適用するか
    input_session_ids = session_ids if split_by_gap and split_scope in ('input', 'both') else None
    graph_session_ids = session_ids if split_by_gap and split_scope in ('graph', 'both') else None
    if split_by_gap: print(f"Trajectories (gap > {session_gap_hours:g}h で分割, 適用先: {split_scope}): {num_sessions}")
    train_dataset = CheckinDataset(raw_data, seq_len=Config['SEQ_LEN'], num_pois=num_pois, usage='train', session_ids=input_session_ids)
    val_dataset = CheckinDataset(raw_data, seq_len=Config['SEQ_LEN'], num_pois=num_pois, usage='validation', session_ids=input_session_ids)
    test_dataset = CheckinDataset(raw_data, seq_len=Config['SEQ_LEN'], num_pois=num_pois, usage='test', session_ids=input_session_ids)
    print(f"Samples: Train={len(train_dataset)}, Valid={len(val_dataset)}, Test={len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=Config['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config['BATCH_SIZE'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config['BATCH_SIZE'], shuffle=False)
    
    model = TWTransNet(
        num_users, num_pois, Config['EMBED_DIM'], 24, num_now_weathers, num_day_weathers, num_month_weathers, 
        Config['DROPOUT'], Config['GAMMA'], use_season=Config['USE_SEASON'], use_gnn=Config['USE_GNN'], 
        is_baseline=existing, skip_user_agg=Config['SKIP_USER_AGG'], skip_poi_agg=Config['SKIP_POI_AGG'],
        use_enhanced_traj=Config['USE_ENHANCED_TRAJ'], enhanced_traj_k=Config['ENHANCED_TRAJ_K'],
        use_time=Config['USE_TIME'], use_weather=Config['USE_WEATHER'], seq_len=Config['SEQ_LEN'],
        dist_norm=dist_norm
    ).to(device)
    if dist_norm != 'legacy':
        d_s_max, d_t_max = compute_dist_scale(train_dataset, Config['SEQ_LEN'], pct=dist_pct)
        model.st_attention.set_dist_scale(d_s_max, d_t_max)
        print(f"Distance normalization: {dist_norm} | D_s({dist_pct:g}%)={d_s_max:.3f} km, D_t({dist_pct:g}%)={d_t_max:.3f} h")
    # --auto_lambda: λ を固定値ではなく学習可能な重み（AutoLossWeight）で自動決定する。--lamda は初期値として使う
    loss_weighter = AutoLossWeight(init_lambda=lamda_weight, max_lambda=auto_lambda_max).to(device) if (auto_lambda and not skip_trans_loss) else None
    params = list(model.parameters()) + (list(loss_weighter.parameters()) if loss_weighter is not None else [])
    optimizer = optim.Adam(params, lr=Config['LR'])
    if loss_weighter is not None: print(f"Auto lambda enabled (uncertainty weighting), initial lambda = {loss_weighter.effective_lambda():.4f}")
    
    # グラフからのデータリークを防ぐため、検証・テストの正解を除外したデータでグラフを構築
    # 除外対象はデータセットと同じ単位で決める（セッション分割時は各セッションの末尾2件）
    # leak=True の場合は比較実験用に、全訪問（検証・テストの正解を含む）でグラフを構築する
    val_targets, test_targets = {}, {}
    for u_id, traj in raw_data.items():
        sids = input_session_ids.get(u_id) if input_session_ids else None
        val_targets[u_id], test_targets[u_id] = get_eval_target_indices(len(traj), sids)

    def build_graph_data(excluded):
        graph_data, graph_sids = {}, {}
        for u_id, traj in raw_data.items():
            keep = [i for i in range(len(traj)) if leak or i not in excluded[u_id]]
            graph_data[u_id] = [traj[i] for i in keep]
            if graph_session_ids: graph_sids[u_id] = [graph_session_ids[u_id][i] for i in keep]
        return graph_data, (graph_sids if graph_session_ids else None)

    # 学習・検証時は検証用とテスト用の正解をどちらも除外する
    graph_train_data, graph_train_sids = build_graph_data({u: val_targets[u] | test_targets[u] for u in raw_data})
    edges = build_interaction_graph(graph_train_data, device=device, session_ids=graph_train_sids)

    # テスト時は検証用の正解をグラフに戻し、テスト用の正解のみ除外する
    graph_test_data, graph_test_sids = build_graph_data(test_targets)
    test_edges = build_interaction_graph(graph_test_data, device=device, session_ids=graph_test_sids)
    # CAAI 強化軌跡グラフ: 遷移近傍は学習用グラフ（val/test の正解を除外）の POI 遷移から作る
    if use_enhanced_traj: model.set_transition_neighbors(*build_transition_neighbors(edges, num_pois))
    trans_sampler = TransNegSampler(train_dataset, num_pois, model, device=device) if (trans_neg == 'constrained' and not skip_trans_loss) else None
    if leak: print("WARNING: leak mode - 検証・テストの正解を含む全訪問でグラフを構築しています（比較実験用）")

    checkpoint_path = f"checkpoint_{city}_lambda{lamda_weight}" + \
                      ("_existing" if existing else 
                       ("_noseason" if not use_season else "") + 
                       ("_nognn" if not use_gnn else "") + 
                       ("_no_user_agg" if skip_user_agg else "") + 
                       ("_no_poi_agg" if skip_poi_agg else "") + 
                       ("_no_trans_loss" if skip_trans_loss else "")) + \
                      (f"_enhanced_traj_k{enhanced_traj_k}" if use_enhanced_traj else "") + \
                      ("_noweather" if not use_weather else "") + \
                      ("_notime" if not use_time else "") + \
                      (f"_gap{session_gap_hours:g}h_{split_scope}" if split_by_gap else "") + \
                      ("_leak" if leak else "") + \
                      (f"_distlog{dist_pct:g}" if dist_norm == 'log' else "") + \
                      (("_autolambda" + (f"_max{auto_lambda_max:g}" if auto_lambda_max else "")) if loss_weighter is not None else "") + \
                      ("_ce" if rec_loss == 'ce' else "_sce" if rec_loss == 'sampled_ce' else "") + \
                      (f"_seq{seq_len}" if seq_len != 10 else "") + \
                      ("_selacc" if select_metric == 'acc_mean' else "") + \
                      ("_tnc" if trans_sampler is not None else "") + ".pth"
    best_val_ndcg10, best_val_acc1, best_val_acc5, best_val_acc10, best_val_mrr = -1, -1, -1, -1, -1
    best_model_state, start_epoch = None, 0
    # ベストモデルの選択基準: ndcg10 = 検証 nDCG@10（従来）, acc_mean = 検証 Acc@1/5/10/20 の平均（STAN 公開実装と同じ）
    select_name = 'nDCG@10' if select_metric == 'ndcg10' else 'mean(Acc@1,5,10,20)'
    
    if resume and os.path.exists(checkpoint_path):
        cp = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(cp['model_state_dict']); optimizer.load_state_dict(cp['optimizer_state_dict'])
        start_epoch = cp['epoch'] + 1
        best_val_ndcg10 = cp.get('best_val_score', cp.get('best_val_ndcg10', -1))
        best_val_acc1 = cp.get('best_val_acc1', -1)
        best_val_acc5 = cp.get('best_val_acc5', -1)
        best_val_acc10 = cp.get('best_val_acc10', -1)
        best_val_mrr = cp.get('best_val_mrr', -1)
        best_model_state = cp.get('best_model_state')
        if loss_weighter is not None and cp.get('loss_weighter_state') is not None:
            loss_weighter.load_state_dict(cp['loss_weighter_state'])
        print(f"Resuming from epoch {start_epoch} (Best Val {select_name}: {best_val_ndcg10:.4f})")

    for epoch in range(start_epoch, Config['EPOCHS']):

        # Train
        loss, rec_l, trans_l, acc1, acc5, acc10, mrr_val, ndcg10, acc20 = train_epoch(model, train_loader, optimizer, num_pois, 
                                                                       edges, poi_coords, lambda_weight=Config['LAMBDA_WEIGHT'], device=device, skip_poi_agg=Config['SKIP_POI_AGG'], skip_trans_loss=Config['SKIP_TRANS_LOSS'], loss_weighter=loss_weighter, rec_loss_type=Config['REC_LOSS'], trans_sampler=trans_sampler)
        print(f"Epoch {epoch+1}/{Config['EPOCHS']} [Train] | Loss: {loss:.4f} | Rec: {rec_l:.4f} | Trans: {trans_l:.4f} "
              f"| Acc@1: {acc1:.4f} | Acc@5: {acc5:.4f} | Acc@10: {acc10:.4f} | Acc@20: {acc20:.4f} | MRR: {mrr_val:.4f} | nDCG@10: {ndcg10:.4f}")
        if loss_weighter is not None:
            s_rec, s_trans = loss_weighter.log_vars.detach().cpu().tolist()
            print(f"  >>> Auto lambda: {loss_weighter.effective_lambda():.4f} (s_rec={s_rec:.4f}, s_trans={s_trans:.4f})")
        
        # Validation
        v_loss, v_rec_l, v_trans_l, v_acc1, v_acc5, v_acc10, v_mrr_val, v_ndcg10, v_acc20 = evaluate(model, val_loader, num_pois, 
                                                                                  edges, poi_coords, lambda_weight=Config['LAMBDA_WEIGHT'], device=device, skip_poi_agg=Config['SKIP_POI_AGG'], skip_trans_loss=Config['SKIP_TRANS_LOSS'], loss_weighter=loss_weighter, rec_loss_type=Config['REC_LOSS'], trans_sampler=trans_sampler)
        print(f"Epoch {epoch+1}/{Config['EPOCHS']} [Valid] | Loss: {v_loss:.4f} | Rec: {v_rec_l:.4f} | Trans: {v_trans_l:.4f} "
              f"| Acc@1: {v_acc1:.4f} | Acc@5: {v_acc5:.4f} | Acc@10: {v_acc10:.4f} | Acc@20: {v_acc20:.4f} | MRR: {v_mrr_val:.4f} | nDCG@10: {v_ndcg10:.4f}")
        
        # Save Best Model based on the validation selection metric
        v_score = v_ndcg10 if select_metric == 'ndcg10' else float(np.mean([v_acc1, v_acc5, v_acc10, v_acc20]))
        if v_score > best_val_ndcg10:
            best_val_ndcg10 = v_score
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  >>> Best Valid {select_name} updated: {best_val_ndcg10:.4f}")
        else:
            print(f"  >>> Best Valid {select_name} not updated: {best_val_ndcg10:.4f}")

        # --- Save Checkpoint ---
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_ndcg10': best_val_ndcg10,  # 互換用（中身は best_val_score と同じ）
            'best_val_score': best_val_ndcg10, 'select_metric': select_metric,
            'best_model_state': best_model_state,
            'loss_weighter_state': loss_weighter.state_dict() if loss_weighter is not None else None,
            'poi_list': POI_list,    # 位置 i が POI index i+1 の aoi_id
            'poi_names': poi_names   # POI index -> 観光地名 (matched_aoi)
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"  >>> Checkpoint saved to {checkpoint_path}")
    
    if best_model_state: model.load_state_dict(best_model_state)
    if use_enhanced_traj: model.set_transition_neighbors(*build_transition_neighbors(test_edges, num_pois))
    t = evaluate(model, test_loader, num_pois, test_edges, poi_coords, lambda_weight=Config['LAMBDA_WEIGHT'], device=device, skip_poi_agg=Config['SKIP_POI_AGG'], skip_trans_loss=Config['SKIP_TRANS_LOSS'], loss_weighter=loss_weighter, rec_loss_type=Config['REC_LOSS'], trans_sampler=trans_sampler)
    print(f"Final Test Result | Loss: {t[0]:.4f} | Acc@1: {t[3]:.4f} Acc@5: {t[4]:.4f} Acc@10: {t[5]:.4f} Acc@20: {t[8]:.4f} MRR: {t[6]:.4f} nDCG@10: {t[7]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lamda', type=float, default=0.5); parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--no_season', action='store_true'); parser.add_argument('--no_gnn', action='store_true')
    parser.add_argument('--no_weather', action='store_true', help='単一の気象情報(weather_now_embedding)を使わない')
    parser.add_argument('--no_time', action='store_true', help='時間情報(time_embedding)を使わない')
    parser.add_argument('--no_user_agg', action='store_true')
    parser.add_argument('--no_poi_agg', action='store_true')
    parser.add_argument('--no_trans_loss', action='store_true')
    parser.add_argument('--rec_loss', type=str, default='sampled_ce', choices=['sampled_ce', 'neg', 'ce'],
                        help='推薦損失: sampled_ce=STAN 公開実装（正解+負例10個の softmax CE）, '
                             'neg=論文の式（STAN Eq11 / CAAI Eq14 の σ 形式）, ce=全候補 softmax の CrossEntropy（旧版）')
    parser.add_argument('--seq_len', type=int, default=10, help='入力系列の最大長（直近 seq_len 件）。STAN / CAAI は 100')
    parser.add_argument('--select_metric', type=str, default='ndcg10', choices=['ndcg10', 'acc_mean'],
                        help='ベストモデルの選択基準: ndcg10=検証 nDCG@10, acc_mean=検証 Acc@1/5/10/20 の平均（STAN 公開実装と同じ）')
    parser.add_argument('--auto_lambda', action='store_true', help='λ を学習可能な重み（不確実性重み付け, Kendall+18）で自動決定する。--lamda は初期値')
    parser.add_argument('--auto_lambda_max', type=float, default=None, help='--auto_lambda 時の実効 λ の上限（未指定なら上限なし）')
    parser.add_argument('--enhance_traj', action='store_true')
    parser.add_argument('--enhance_k', type=int, default=6, help='強化軌跡グラフで各POIに残す遷移近傍数 K（CAAI の最良値 K=6）')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--city', type=str, default='NYC'); parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--existing', action='store_true')
    parser.add_argument('--session_gap', type=float, default=0, help='同一ユーザでこの時間(h)を超えて間隔が空いたら別の軌跡とみなす（0以下で分割しない＝従来どおり）')
    parser.add_argument('--leak', action='store_true', help='検証・テストの正解をグラフから除外しない（リークあり、比較実験用）')
    parser.add_argument('--dist_norm', type=str, default='legacy', choices=['legacy', 'log'],
                        help='時空間ペナルティ: legacy=w·log(1+d)（従来式）, log=softplus(θ)·min(log(1+d)/log(1+D),1)（D は学習データのパーセンタイル）')
    parser.add_argument('--dist_pct', type=float, default=95, help='正規化の上限 D に使うパーセンタイル')
    parser.add_argument('--split_scope', type=str, default='both', choices=['input', 'graph', 'both'],
                        help='軌跡の区切りの適用先: input=入力系列のみ, graph=グラフの遷移エッジのみ(--enhance_traj時のみ有効), both=両方')
    parser.add_argument('--trans_neg', type=str, default='constrained', choices=['constrained', 'uniform'],
                        help='翻訳損失の負例 tail: constrained=CAAI の制約 (a) 正解と異なる (b) 同ユーザ・同文脈 r で遷移済みの POI を除く, uniform=制約なし一様（旧版）')
    args = parser.parse_args()
    main(args.lamda, city=args.city, resume=not args.fresh, seed=args.seed, use_season=not args.no_season, use_gnn=not args.no_gnn, existing=args.existing, skip_user_agg=args.no_user_agg, skip_poi_agg=args.no_poi_agg, skip_trans_loss=args.no_trans_loss, use_enhanced_traj=args.enhance_traj, enhanced_traj_k=args.enhance_k, epochs=args.epochs, session_gap_hours=args.session_gap, split_scope=args.split_scope, leak=args.leak, use_time=not args.no_time, use_weather=not args.no_weather, dist_norm=args.dist_norm, dist_pct=args.dist_pct, auto_lambda=args.auto_lambda, auto_lambda_max=args.auto_lambda_max, rec_loss=args.rec_loss, seq_len=args.seq_len, select_metric=args.select_metric, trans_neg=args.trans_neg)

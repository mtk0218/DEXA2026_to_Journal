import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def get_session_groups(traj_len, sids=None):
    """traj 上のインデックスをセッションごとにまとめる（sids が None ならユーザ全体で1セッション）"""
    if sids is None:
        return [list(range(traj_len))]
    session_groups = {}
    for idx, sid in enumerate(sids):
        session_groups.setdefault(sid, []).append(idx)
    return list(session_groups.values())

def get_eval_target_indices(traj_len, sids=None):
    """
    CheckinDataset と同じ規則で、検証・テストの正解になる traj 上のインデックスを返す。
    グラフから正解を除外する際に使う。
    """
    val_indices, test_indices = set(), set()
    for indices in get_session_groups(traj_len, sids):
        if len(indices) < 4:
            continue  # CheckinDataset と同様、短いセッションは評価対象にならない
        val_indices.add(indices[-2])
        test_indices.add(indices[-1])
    return val_indices, test_indices

class CheckinDataset(Dataset):
    """
    Dataset for TWTransNet.

    session_ids が None の場合（分割なし）:
        ユーザ全軌跡を1セッションとして扱う。
        train: index 1 ~ L-3 をターゲット
        val:   index L-2 をターゲット
        test:  index L-1 をターゲット

    session_ids が指定されている場合（セッション分割あり）:
        ユーザの軌跡をセッションごとに分割し、
        各セッション単位で train/val/test を生成する。
        train: 各セッションの index 1 ~ Ls-3 をターゲット
        val:   各セッションの index Ls-2 をターゲット
        test:  各セッションの index Ls-1 をターゲット
        ※ Ls < 4 のセッションはスキップ
        入力系列は必ず同一セッション内に限定する。
    """
    def __init__(self, user_trajectories, seq_len=5, num_pois=None, usage='train', session_ids=None):
        self.user_trajectories = user_trajectories
        self.session_ids = session_ids
        self.seq_len = seq_len
        self.num_pois = num_pois
        self.usage = usage

        self.data_samples = []
        self._process_trajectories()

    def _process_trajectories(self):
        for user_id, traj in self.user_trajectories.items():
            sids = self.session_ids.get(user_id) if self.session_ids else None

            # 分割なしならユーザ全体を1セッション、分割ありならセッションごとに処理
            for indices in get_session_groups(len(traj), sids):
                # セッション内のグローバルインデックスリストを渡す
                self._process_session(user_id, traj, global_indices=indices)

    def _process_session(self, user_id, traj, global_indices):
        """
        global_indices: セッション内のtraj上のインデックスリスト（昇順）
        セッション内でのtrain/val/testを生成する。
        """
        Ls = len(global_indices)
        if Ls < 4:
            return  # セッションが短すぎてval/testが作れない

        # セッション内での相対位置でtargetを決定
        if self.usage == 'train':
            rel_target_indices = range(1, Ls - 2)
        elif self.usage == 'validation':
            rel_target_indices = [Ls - 2]
        elif self.usage == 'test':
            rel_target_indices = [Ls - 1]
        else:
            return

        for rel_i in rel_target_indices:
            global_i = global_indices[rel_i]
            target = traj[global_i]

            # 入力系列: セッション内の直前 seq_len 件
            rel_start = max(0, rel_i - self.seq_len)
            input_global_indices = global_indices[rel_start:rel_i]
            input_seq = [traj[g] for g in input_global_indices]

            if len(input_seq) == 0:
                continue

            self.data_samples.append({
                'user_id': user_id,
                'input_seq': input_seq,
                'target': target,
            })

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        user_id, input_seq, target = sample['user_id'], sample['input_seq'], sample['target']

        # Unpack input sequence
        pois         = [x[0] for x in input_seq]
        times        = [x[1] for x in input_seq]
        now_weathers = [x[2] for x in input_seq]
        lats         = [x[3] for x in input_seq]
        lons         = [x[4] for x in input_seq]

        has_full = len(input_seq[0]) >= 10
        if has_full:
            yests   = [x[5] for x in input_seq]
            todays  = [x[6] for x in input_seq]
            toms    = [x[7] for x in input_seq]
            months  = [x[8] for x in input_seq]
            seasons = [x[9] for x in input_seq]
        else:
            yests = todays = toms = months = seasons = [0] * len(input_seq)

        # Padding
        curr_len = len(pois)
        if curr_len < self.seq_len:
            pad = self.seq_len - curr_len
            pois         = [0]   * pad + pois
            times        = [0.0] * pad + times
            now_weathers = [0]   * pad + now_weathers
            lats         = [0.0] * pad + lats
            lons         = [0.0] * pad + lons
            yests        = [0]   * pad + yests
            todays       = [0]   * pad + todays
            toms         = [0]   * pad + toms
            months       = [0]   * pad + months
            seasons      = [0]   * pad + seasons
            mask = [0] * pad + [1] * curr_len
        else:
            offset = curr_len - self.seq_len
            pois         = pois[offset:]
            times        = times[offset:]
            now_weathers = now_weathers[offset:]
            lats         = lats[offset:]
            lons         = lons[offset:]
            yests        = yests[offset:]
            todays       = todays[offset:]
            toms         = toms[offset:]
            months       = months[offset:]
            seasons      = seasons[offset:]
            mask = [1] * self.seq_len

        # Unpack target & head
        target_poi, target_time, target_now_w, target_lat, target_lon = target[:5]
        head_poi, head_time, head_now_w = input_seq[-1][:3]

        if has_full:
            tar_y, tar_t, tar_tm, tar_m, tar_s = target[5:10]
            hed_y, hed_t, hed_tm, hed_m, hed_s = input_seq[-1][5:10]
        else:
            tar_y = tar_t = tar_tm = tar_m = tar_s = 0
            hed_y = hed_t = hed_tm = hed_m = hed_s = 0

        return {
            'user_id':                  torch.tensor(user_id,       dtype=torch.long),
            'traj_poi':                 torch.tensor(pois,          dtype=torch.long),
            'traj_time':                torch.tensor(times,         dtype=torch.float),
            'traj_now_weather':         torch.tensor(now_weathers,  dtype=torch.long),
            'traj_lat':                 torch.tensor(lats,          dtype=torch.float),
            'traj_lon':                 torch.tensor(lons,          dtype=torch.float),
            'traj_yesterday_weather':   torch.tensor(yests,         dtype=torch.long),
            'traj_today_weather':       torch.tensor(todays,        dtype=torch.long),
            'traj_tomorrow_weather':    torch.tensor(toms,          dtype=torch.long),
            'traj_month_weather':       torch.tensor(months,        dtype=torch.long),
            'traj_season':              torch.tensor(seasons,       dtype=torch.long),
            'mask':                     torch.tensor(mask,          dtype=torch.long),

            'head_poi':                 torch.tensor(head_poi,      dtype=torch.long),
            'target_poi':               torch.tensor(target_poi,    dtype=torch.long),
            'target_time':              torch.tensor(target_time,   dtype=torch.float),
            'target_now_weather':       torch.tensor(target_now_w,  dtype=torch.long),
            'target_lat':               torch.tensor(target_lat,    dtype=torch.float),
            'target_lon':               torch.tensor(target_lon,    dtype=torch.float),
            'target_yesterday_weather': torch.tensor(tar_y,         dtype=torch.long),
            'target_today_weather':     torch.tensor(tar_t,         dtype=torch.long),
            'target_tomorrow_weather':  torch.tensor(tar_tm,        dtype=torch.long),
            'target_month_weather':     torch.tensor(tar_m,         dtype=torch.long),
            'target_season':            torch.tensor(tar_s,         dtype=torch.long),

            'head_time':                torch.tensor(head_time,     dtype=torch.float),
            'head_now_weather':         torch.tensor(head_now_w,    dtype=torch.long),
            'head_yesterday_weather':   torch.tensor(hed_y,         dtype=torch.long),
            'head_today_weather':       torch.tensor(hed_t,         dtype=torch.long),
            'head_tomorrow_weather':    torch.tensor(hed_tm,        dtype=torch.long),
            'head_month_weather':       torch.tensor(hed_m,         dtype=torch.long),
            'head_season':              torch.tensor(hed_s,         dtype=torch.long),

            # Legacy/Baseline Aliases
            'traj_weather':   torch.tensor(now_weathers, dtype=torch.long),
            'target_weather': torch.tensor(target_now_w, dtype=torch.long),
            'head_weather':   torch.tensor(head_now_w,   dtype=torch.long),
        }
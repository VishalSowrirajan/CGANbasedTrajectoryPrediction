import logging
import os
import math

import numpy as np
from torch.utils.data import DataLoader

import torch
from torch.utils.data import Dataset
from constants import *


def data_loader(path, metric):
    dset = TrajectoryDataset(
        path,
        metric)

    loader = DataLoader(
        dset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=seq_collate)
    return dset, loader


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, loss_mask_list, obs_ped_abs_speed,
     pred_ped_abs_speed, obs_label, pred_label) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    obs_ped_abs_speed = torch.cat(obs_ped_abs_speed, dim=0).permute(2, 0, 1)
    pred_ped_abs_speed = torch.cat(pred_ped_abs_speed, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)
    loss_mask = torch.cat(loss_mask_list, dim=0)

    obs_label = torch.cat(obs_label, dim=0).permute(2, 0, 1)
    pred_label = torch.cat(pred_label, dim=0).permute(2, 0, 1)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, loss_mask, seq_start_end, obs_ped_abs_speed,
        pred_ped_abs_speed, obs_label, pred_label
    ]

    return tuple(out)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def read_file(_path, delim='\t'):
    data = []
    i = 0
    with open(_path, 'r') as f:
        for line in f:
            if i == 0:
                i += 1
                continue
            line = line.strip().split(',')
            line = [i for i in line]
            data.append(line)
    return np.asarray(data)


def get_min_max_speed_labels(seq_len, all_files):
    all_speed, av_speed, other_speed, agent_speed, city_label = [], [], [], [], []

    for path in all_files:
        data = read_file(path, ' ')
        frames = np.unique(data[:, 0]).tolist()
        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :5])
        num_sequences = int(math.ceil((len(frames) - seq_len + 1)))

        for idx in range(0, num_sequences):
            curr_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0)
            ped_in_curr_seq = np.unique(curr_seq_data[:, 1])
            for _, obj_id in enumerate(ped_in_curr_seq):
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == obj_id, :]
                label = curr_ped_seq[0, 2]
                city_labels = curr_ped_seq[0, -1]
                pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                if pad_end - pad_front != 20:
                    continue
                if len(curr_ped_seq[:, 0]) != 20:
                    continue
                curr_ped_x_axis_new = [0.0] + [np.square(float(t) - float(s)) for s, t in
                                               zip(curr_ped_seq[:, 3], curr_ped_seq[1:, 3])]
                curr_ped_y_axis_new = [0.0] + [np.square(float(t) - float(s)) for s, t in
                                               zip(curr_ped_seq[:, 4], curr_ped_seq[1:, 4])]
                #city_label.append(city_labels)

                curr_ped_dist = np.sqrt(np.add(curr_ped_x_axis_new, curr_ped_y_axis_new))
                curr_ped_dist = np.around(curr_ped_dist, decimals=1)
                all_speed.append(curr_ped_dist)
                if label == 'AV':
                    av_speed.append(curr_ped_dist)
                if label == 'AGENT':
                    agent_speed.append(curr_ped_dist)
                if label == 'OTHERS':
                    other_speed.append(curr_ped_dist)

    all_speed = np.array(all_speed).reshape(-1, 1)
    other_speed = np.array(other_speed).reshape(-1, 1)
    agent_speed = np.array(agent_speed).reshape(-1, 1)
    av_speed = np.array(av_speed).reshape(-1, 1)

    uniqueallspeed, countsallspeed = np.unique(all_speed, return_counts=True)
    all_count = dict(zip(uniqueallspeed, countsallspeed))
    print(all_count)

    uniqueother_speed, countsother_speed = np.unique(other_speed, return_counts=True)
    other_count = dict(zip(uniqueother_speed, countsother_speed))
    print(other_count)

    uniqueagent_speed, countsagent_speed = np.unique(agent_speed, return_counts=True)
    agent_count = dict(zip(uniqueagent_speed, countsagent_speed))
    print(agent_count)

    uniqueav_speed, countsav_speed = np.unique(av_speed, return_counts=True)
    av_count = dict(zip(uniqueav_speed, countsav_speed))
    print(av_count)
    #city_label = np.array(city_label).reshape(-1, 1)
    #labels = np.unique(city_label)
    max_speed = np.amax(all_speed)
    min_speed = np.min(all_speed)

    max_other_speed = np.amax(other_speed)
    min_other_speed = np.min(other_speed)

    max_agent_speed = np.amax(agent_speed)
    min_agent_speed = np.min(agent_speed)

    max_av_speed = np.amax(av_speed)
    min_av_speed = np.min(av_speed)

    return max_speed, min_speed


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, metric=0
    ):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        SEQ_LEN = OBS_LEN + PRED_LEN
        self.train_or_test = metric

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        ped_abs_speed = []
        obj_label = []
        loss_mask_list = []
        for path in all_files:
            data = read_file(path, '\t')
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - SEQ_LEN + 1)))
            #min, max = get_min_max_speed_labels(SEQ_LEN, all_files)

            for idx in range(0, num_sequences + 1):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + SEQ_LEN], axis=0)

                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), SEQ_LEN))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, SEQ_LEN))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, SEQ_LEN))
                _curr_ped_abs_speed = np.zeros((len(peds_in_curr_seq), SEQ_LEN))
                _curr_obj_label = np.zeros((len(peds_in_curr_seq), SEQ_LEN))
                num_peds_considered = 0

                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    label = curr_ped_seq[0, 2]
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != SEQ_LEN:
                        continue
                    if len(curr_ped_seq[:, 0]) != SEQ_LEN:
                        continue
                    curr_ped_x_axis_new = [0.0] + [np.square(float(t) - float(s)) for s, t in
                                                   zip(curr_ped_seq[:, 3], curr_ped_seq[1:, 3])]
                    curr_ped_y_axis_new = [0.0] + [np.square(float(t) - float(s)) for s, t in
                                                   zip(curr_ped_seq[:, 4], curr_ped_seq[1:, 4])]

                    curr_ped_dist = np.sqrt(np.add(curr_ped_x_axis_new, curr_ped_y_axis_new))
                    # Since each frame is taken with an interval of 0.1, we divide the distance with 0.1 to get speed
                    curr_ped_abs_speed = curr_ped_dist
                    curr_ped_abs_speed = [sigmoid(x) for x in curr_ped_abs_speed]
                    curr_ped_abs_speed = np.around(curr_ped_abs_speed, decimals=5)
                    curr_ped_abs_speed = np.transpose(curr_ped_abs_speed)

                    if label == 'AV':  # Pedestrians
                        embedding_label = 0.1
                    elif label == 'OTHERS':  # Vehicles
                        embedding_label = 0.2
                    elif label == 'AGENT':  # Vehicles
                        embedding_label = 0.3

                    curr_ped_seq = np.transpose(curr_ped_seq[:, 3:5])
                    curr_ped_seq = curr_ped_seq.astype(float)
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    _curr_ped_abs_speed[_idx, pad_front:pad_end] = curr_ped_abs_speed
                    _curr_obj_label[_idx, pad_front:pad_end] = embedding_label
                    num_peds_considered += 1

                if num_peds_considered > 1:
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    ped_abs_speed.append(_curr_ped_abs_speed[:num_peds_considered])
                    obj_label.append(_curr_obj_label[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        ped_abs_speed = np.concatenate(ped_abs_speed, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        ped_abs_speed = torch.from_numpy(ped_abs_speed).type(torch.float)
        obj_label = np.concatenate(obj_label, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :OBS_LEN]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, OBS_LEN:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :OBS_LEN]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, OBS_LEN:]).type(torch.float)

        self.obs_ped_abs_speed = ped_abs_speed[:, :OBS_LEN].unsqueeze(dim=1).type(torch.float)
        self.pred_ped_abs_speed = ped_abs_speed[:, OBS_LEN:].unsqueeze(dim=1).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)

        self.obs_obj_label = torch.from_numpy(obj_label[:, :OBS_LEN]).unsqueeze(dim=1).type(torch.float)
        self.pred_obj_label = torch.from_numpy(obj_label[:, OBS_LEN:]).unsqueeze(dim=1).type(torch.float)

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.loss_mask[start:end, :], self.obs_ped_abs_speed[start:end, :],
            self.pred_ped_abs_speed[start:end, :], self.obs_obj_label[start:end, :],
            self.pred_obj_label[start:end, :]
        ]
        return out
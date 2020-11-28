import torch
import random
import torch.nn as nn

from constants import *


def bce_loss(input, target):
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake):
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode='average'):
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    seq_len, count, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss), count
    elif mode == 'raw':
        return loss, count


def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode='sum'):
    loss = pred_pos_gt - pred_pos
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def mean_speed_error(real_speed, fake_speed):
    # Mean speed loss over all timesteps - Used only for feedback and not for training the model
    speed_loss = torch.abs(real_speed - fake_speed)
    add_loss = torch.sum(speed_loss, dim=1)
    add_loss_1 = torch.sum(add_loss)
    return add_loss_1


def final_speed_error(real_speed, fake_speed):
    # Final traj speed loss - Used only for feedback and not for training the model
    speed_loss = torch.abs(real_speed - fake_speed)
    add_loss_1 = torch.sum(speed_loss)
    return add_loss_1


def relative_to_abs(rel_traj, start_pos):
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def displacement_error_test(pred_traj, pred_traj_gt, seq_start_end, label, mode='sum'):
    weighted_ade = []
    pdist = nn.PairwiseDistance(p=2)
    seq_len, count, _ = pred_traj.size()
    pred_traj_gt = pred_traj_gt.permute(1, 0, 2).reshape(-1, 2)
    pred_traj = pred_traj.permute(1, 0, 2).reshape(-1, 2)
    euclideanerror = pdist(pred_traj_gt, pred_traj)

    label_plus_error = torch.cat([euclideanerror.reshape(PRED_LEN, -1, 1), label], dim=2)

    for (start, end) in seq_start_end:
        curr_ped_loss, curr_veh_loss, curr_cyc_loss = [], [], []
        curr_label_error = label_plus_error[:, start:end, :]
        for label_error in curr_label_error:
            for a in label_error:
                if a[-1] == 0.1:
                    curr_veh_loss.append(a[0])
                if a[-1] == 0.3:
                    curr_ped_loss.append(a[0])
                if a[-1] == 0.4:
                    curr_cyc_loss.append(a[0])
        if len(curr_ped_loss) != 0:
            ped_loss = sum(curr_ped_loss) / len(curr_ped_loss)
        else:
            ped_loss = 0
        if len(curr_veh_loss) != 0:
            veh_loss = sum(curr_veh_loss) / len(curr_veh_loss)
        else:
            veh_loss = 0
        if len(curr_cyc_loss) != 0:
            cyc_loss = sum(curr_cyc_loss) / len(curr_cyc_loss)
        else:
            cyc_loss = 0

        ade = PEDESTRIAN_COE * ped_loss + BICYCLE_COE * cyc_loss + VEHICLE_COE * veh_loss
        weighted_ade.append(ade)

    a = sum(weighted_ade) / len(weighted_ade)
    return sum(weighted_ade) / len(weighted_ade)


def testing_metric(pred_traj, pred_traj_gt, seq_start_end, label, mode='sum'):
    loss_fun = []
    pdist = nn.PairwiseDistance(p=2)
    seq_len, count, _ = pred_traj.size()
    pred_traj_gt = pred_traj_gt.permute(1, 0, 2).reshape(-1, 2)
    pred_traj = pred_traj.permute(1, 0, 2).reshape(-1, 2)
    euclideanerror = pdist(pred_traj_gt, pred_traj)

    label_plus_error = torch.cat([euclideanerror.reshape(PRED_LEN, -1, 1), label], dim=2)

    for (start, end) in seq_start_end:
        curr_ped_loss, curr_veh_loss, curr_cyc_loss = [], [], []
        veh_count, ped_count, cyc_count = 0, 0, 0
        curr_label_error = label_plus_error[:, start:end, :]
        for a in curr_label_error.reshape(-1, 2):
            if a[-1] == 0.1:
                curr_veh_loss.append(a[0] * VEHICLE_COE)
                veh_count += 1
            if a[-1] == 0.3:
                curr_ped_loss.append(a[0] * PEDESTRIAN_COE)
                ped_count += 1
            if a[-1] == 0.4:
                curr_cyc_loss.append(a[0] * BICYCLE_COE)
                cyc_count += 1

        if len(curr_ped_loss) != 0:
            _ped = torch.stack(curr_ped_loss, dim=0).reshape(int(ped_count/PRED_LEN), PRED_LEN)
            _ped_loss = torch.sum(_ped, dim=1)
        if len(curr_veh_loss) != 0:
            _veh = torch.stack(curr_veh_loss, dim=0)
            _veh = _veh.reshape(int(veh_count/PRED_LEN), PRED_LEN)
            _veh_loss = torch.sum(_veh, dim=1)
        if len(curr_cyc_loss) != 0:
            _cyc = torch.stack(curr_cyc_loss, dim=0).reshape(int(cyc_count/PRED_LEN), PRED_LEN)
            _cyc_loss = torch.sum(_cyc, dim=1)

    loss_fun = torch.cat(loss_fun, dim=0)
    return loss_fun.view(-1, 1)
import gc
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from constants import *

from trajectories import data_loader
from utils import gan_g_loss, gan_d_loss, l2_loss, mean_speed_error, \
    final_speed_error, displacement_error, final_displacement_error, relative_to_abs, bce_loss, displacement_error_test

from models import TrajectoryGenerator, TrajectoryDiscriminator

torch.backends.cudnn.benchmark = True


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def main():
    train_metric = 0
    print("Process Started")
    print("Initializing train dataset")
    train_dset, train_loader = data_loader(TRAIN_DATASET_PATH, train_metric, ' ')
    print("Initializing val dataset")
    val_dset, val_loader = data_loader(VAL_DATASET_PATH, train_metric, '\t')

    iterations_per_epoch = len(train_dset) / BATCH / D_STEPS
    if NUM_EPOCHS:
        NUM_ITERATIONS = int(iterations_per_epoch * NUM_EPOCHS)
        print('TRAIN num iterations per epoch: ', NUM_ITERATIONS)
    iterations_per_epoch = len(val_dset) / BATCH / D_STEPS
    if NUM_EPOCHS:
        NUM_ITERATIONS = int(iterations_per_epoch * NUM_EPOCHS)
        print('VAL num iterations per epoch: ', NUM_ITERATIONS)

    generator = TrajectoryGenerator()

    generator.apply(init_weights)
    generator.type(torch.FloatTensor).train()
    print(generator)

    discriminator = TrajectoryDiscriminator()

    discriminator.apply(init_weights)
    discriminator.type(torch.FloatTensor).train()

    print('Here is the discriminator:')
    print(discriminator)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.RMSprop(generator.parameters(), lr=G_LEARNING_RATE)
    optimizer_d = optim.RMSprop(discriminator.parameters(), lr=D_LEARNING_RATE)

    t, epoch = 0, 0
    checkpoint = {
        'G_losses': defaultdict(list),
        'D_losses': defaultdict(list),
        'g_state': None,
        'g_optim_state': None,
        'd_state': None,
        'd_optim_state': None,
        'g_best_state': None,
        'd_best_state': None
    }
    ade_list, fde_list, wa_ade_list, wa_fde_list, veh_ade_list, veh_fde_list, ped_ade_list, ped_fde_list, \
    cyc_ade_list, cyc_fde_list, avg_speed_error, f_speed_error = [], [], [], [], [], [], [], [], [], [], [], []
    while epoch < NUM_EPOCHS:
        gc.collect()
        d_steps_left, g_steps_left = D_STEPS, G_STEPS
        epoch += 1
        print('Starting epoch {}'.format(epoch))
        for batch in train_loader:
            if d_steps_left > 0:
                losses_d = discriminator_step(batch, generator, discriminator, d_loss_fn, optimizer_d)
                d_steps_left -= 1
            elif g_steps_left > 0:
                losses_g = generator_step(batch, generator, discriminator, g_loss_fn, optimizer_g)
                g_steps_left -= 1

            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if t > 0 and t % CHECKPOINT_EVERY == 0:

                print('t = {} / {}'.format(t + 1, NUM_ITERATIONS))
                for k, v in sorted(losses_d.items()):
                    print('  [D] {}: {:.3f}'.format(k, v))
                for k, v in sorted(losses_g.items()):
                    print('  [G] {}: {:.3f}'.format(k, v))

                print('Checking stats on val ...')
                metrics_val = check_accuracy(val_loader, generator, discriminator, d_loss_fn)
                print('Checking stats on train ...')
                metrics_train = check_accuracy(train_loader, generator, discriminator, d_loss_fn)

                for k, v in sorted(metrics_val.items()):
                    print('  [val] {}: {:.3f}'.format(k, v))
                for k, v in sorted(metrics_train.items()):
                    print('  [train] {}: {:.3f}'.format(k, v))

                #wa_ade_list.append(metrics_train['WSADE'])
                #wa_fde_list.append(metrics_train['WSFDE'])

                ade_list.append(metrics_val['ade'])
                #fde_list.append(metrics_train['fde'])

                # VEHICLE PARAMS
                #veh_ade_list.append(metrics_train['veh_ade'])
                #veh_fde_list.append(metrics_train['veh_fde'])

                # PEDESTRIAN PARAMS
                #ped_ade_list.append(metrics_train['ped_ade'])
                #ped_fde_list.append(metrics_train['ped_fde'])

                # BICYCLIST PARAMS
                #cyc_ade_list.append(metrics_train['cyc_ade'])
                #cyc_fde_list.append(metrics_train['cyc_fde'])

                #avg_speed_error.append(metrics_train['msae'])
                #f_speed_error.append(metrics_train['fse'])

                if metrics_val.get('ade') == min(ade_list) or metrics_val['ade'] < min(ade_list):
                    print('New low for weighted average')
                    checkpoint['g_best_state'] = generator.state_dict()
                #if metrics_val.get('WSFDE') == min(wa_fde_list) or metrics_val['WSFDE'] < min(wa_fde_list):
                #    print('New low for wa_final_disp_error')

                #if metrics_val.get('ade') == min(ade_list) or metrics_val['ade'] < min(ade_list):
                #    print('New low for avg_disp_error')
                #if metrics_val.get('fde') == min(fde_list) or metrics_val['fde'] < min(fde_list):
                #    print('New low for final_disp_error')

                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                torch.save(checkpoint, CHECKPOINT_NAME)
                print('Done.')

            t += 1
            d_steps_left = D_STEPS
            g_steps_left = G_STEPS
            if t >= NUM_ITERATIONS:
                break


def discriminator_step(batch, generator, discriminator, d_loss_fn, optimizer_d):
    """This step is similar to Social GAN Code"""
    batch = [tensor for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed,
     obs_label, pred_label) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out, _ = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                 pred_traj_gt, TRAIN_METRIC, SPEED_TO_ADD, obs_label, pred_label)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
    ped_speed = torch.cat([obs_ped_speed, pred_ped_speed], dim=0)
    label_info = torch.cat([obs_label, pred_label], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, label_info, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, ped_speed, label_info, seq_start_end)

    data_loss = d_loss_fn(scores_real, scores_fake)
    dis_ade = displacement_error_test(pred_traj_fake, pred_traj_gt, seq_start_end, pred_label, mode='sum')
    #dis_ade, _ = displacement_error(pred_traj_fake, pred_traj_gt)
    total_loss = data_loss + dis_ade
    losses['D_data_loss'] = data_loss.item()
    loss += total_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    optimizer_d.step()

    return losses


def generator_step(batch, generator, discriminator, g_loss_fn, optimizer_g):
    """This step is similar to Social GAN Code"""
    batch = [tensor for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed,
     obs_label, pred_label) = batch

    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, OBS_LEN:]

    for _ in range(BEST_K):
        generator_out, _ = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj_gt,
                                     TRAIN_METRIC, SPEED_TO_ADD, obs_label, pred_label)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        if L2_LOSS_WEIGHT > 0:
            g_l2_loss_rel.append(L2_LOSS_WEIGHT * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if L2_LOSS_WEIGHT > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
    ped_speed = torch.cat([obs_ped_speed, pred_ped_speed], dim=0)
    label_info = torch.cat([obs_label, pred_label], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, label_info, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)
    gen_ade = displacement_error_test(pred_traj_fake, pred_traj_gt, seq_start_end, pred_label, mode='sum')

    total_loss = discriminator_loss
    loss += total_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    optimizer_g.step()

    return losses


def check_accuracy(loader, generator, discriminator, d_loss_fn):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    ade_error, veh_disp_error, ped_disp_error, cyc_disp_error = [], [], [], []
    fde_error, veh_f_disp_error, ped_f_disp_error, cyc_f_disp_error = [], [], [], []
    mean_speed_disp_error = []
    final_speed_disp_error = []
    total_traj, veh_total_traj, ped_total_traj, cyc_total_traj  = 0, 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed,
             pred_ped_speed, obs_label, pred_label) = batch

            pred_traj_fake_rel, _ = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                              pred_traj_gt, TRAIN_METRIC, SPEED_TO_ADD, obs_label, pred_label)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
            loss_mask = loss_mask[:, OBS_LEN:]

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )

            weighted_ade = displacement_error_test(pred_traj_fake, pred_traj_gt, seq_start_end, pred_label, mode='sum')
            #general_ade, _ = displacement_error(pred_traj_fake, pred_traj_gt)  # ADE for All
            general_fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])  # FDE for All
            # Get Label-wise Trajectories
            veh_fake, veh_gt, ped_fake, ped_gt, cyc_fake, cyc_gt = get_diff_traj(pred_traj_fake, pred_traj_gt, pred_label)

            # Average Displacement Error
            veh_ade, veh_count = displacement_error(veh_fake, veh_gt)  # ADE for Vehicles
            ped_ade, ped_count = displacement_error(ped_fake, ped_gt)  # ADE for Pedestrians
            cyc_ade, cyc_count = displacement_error(cyc_fake, cyc_gt)  # ADE for Bicycle

            # Final Displacement Error
            veh_fde = final_displacement_error(veh_fake, veh_gt)  # FDE for Vehicles
            ped_fde = final_displacement_error(ped_fake, ped_gt)  # FDE for Pedestrians
            cyc_fde = final_displacement_error(cyc_fake, cyc_gt)  # FDE for Bicycle

            last_pos = obs_traj[-1]
            traj_for_speed_cal = torch.cat([last_pos.unsqueeze(dim=0), pred_traj_fake], dim=0)
            msae = cal_msae(pred_ped_speed, traj_for_speed_cal)
            fse = cal_fse(pred_ped_speed[-1], pred_traj_fake)

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
            ped_speed = torch.cat([obs_ped_speed, pred_ped_speed], dim=0)
            label_info = torch.cat([obs_label, pred_label], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, label_info, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, ped_speed, label_info, seq_start_end)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())

            ade_error.append(weighted_ade.item())
            fde_error.append(general_fde.item())
            veh_disp_error.append(veh_ade.item())
            ped_disp_error.append(ped_ade.item())
            cyc_disp_error.append(cyc_ade.item())
            veh_f_disp_error.append(veh_fde.item())
            ped_f_disp_error.append(ped_fde.item())
            cyc_f_disp_error.append(cyc_fde.item())

            mean_speed_disp_error.append(msae.item())
            final_speed_disp_error.append(fse.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            veh_total_traj += veh_count
            ped_total_traj += ped_count
            cyc_total_traj += cyc_count
            if total_traj >= NUM_SAMPLE_CHECK:
                break

    #veh_wa_ade = VEHICLE_COE * (sum(veh_disp_error) / (veh_total_traj * PRED_LEN))
    #ped_wa_ade = PEDESTRIAN_COE * (sum(ped_disp_error) / (ped_total_traj * PRED_LEN))
    #cyc_wa_ade = BICYCLE_COE * (sum(cyc_disp_error) / (cyc_total_traj * PRED_LEN))
    #veh_wa_fde = VEHICLE_COE * (sum(veh_f_disp_error) / veh_total_traj)
    #ped_wa_fde = PEDESTRIAN_COE * (sum(ped_f_disp_error) / ped_total_traj)
    #cyc_wa_fde = BICYCLE_COE * (sum(cyc_f_disp_error) / cyc_total_traj)
    #metrics['d_loss'] = sum(d_losses) / len(d_losses)
    #metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    #metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum
    #metrics['veh_ade'] = veh_wa_ade
    #metrics['ped_ade'] = ped_wa_ade
    #metrics['cyc_ade'] = cyc_wa_ade
    #metrics['veh_fde'] = veh_wa_fde
    #metrics['ped_fde'] = ped_wa_fde
    #metrics['cyc_fde'] = cyc_wa_fde
    #metrics['WSADE'] = veh_wa_ade + ped_wa_ade + cyc_wa_ade
    #metrics['WSFDE'] = veh_wa_fde + ped_wa_fde + cyc_wa_fde
    metrics['ade'] = sum(ade_error) / len(ade_error)
    #metrics['fde'] = sum(fde_error) / total_traj
    #metrics['msae'] = sum(mean_speed_disp_error) / (total_traj * PRED_LEN)
    #metrics['fse'] = sum(final_speed_disp_error) / total_traj

    generator.train()
    return metrics


def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel, loss_mask):
    g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode='sum')
    g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum')
    return g_l2_loss_abs, g_l2_loss_rel


def get_diff_traj(pred_traj_gt, pred_traj_fake, label):
    vehicle_gt_list, vehicle_pred_list, ped_gt_list, \
    ped_pred_list, cycle_gt_list, cycle_pred_list = [], [], [], [], [], []
    for a,b,c in zip(pred_traj_gt, pred_traj_fake, label):
        for a, b, c in zip(a, b, c):
            if torch.eq(c, 0.1) or torch.eq(c, 0.2):
                vehicle_gt_list.append(a)
                vehicle_pred_list.append(b)
            elif torch.eq(c, 0.3):
                ped_gt_list.append(a)
                ped_pred_list.append(b)
            elif torch.eq(c, 0.4):
                cycle_gt_list.append(a)
                cycle_pred_list.append(b)
    veh_fake, veh_gt = torch.cat(vehicle_pred_list, dim=0).view(PRED_LEN, -1, 2), torch.cat(vehicle_gt_list, dim=0).view(PRED_LEN, -1, 2)
    ped_fake, ped_gt = torch.cat(ped_pred_list, dim=0).view(PRED_LEN, -1, 2), torch.cat(ped_gt_list, dim=0).view(PRED_LEN, -1, 2)
    cyc_fake, cyc_gt = torch.cat(cycle_pred_list).view(PRED_LEN, -1, 2), torch.cat(cycle_gt_list).view(PRED_LEN, -1, 2)
    return veh_fake, veh_gt, ped_fake, ped_gt, cyc_fake, cyc_gt


def cal_msae(real_speed, fake_traj):
    fake_output_speed = fake_speed(fake_traj)
    real_speed = real_speed.permute(1, 0, 2)
    msae = mean_speed_error(real_speed, fake_output_speed)
    return msae


def fake_speed(fake_traj):
    output_speed = []
    for a, b in zip(fake_traj[:, :], fake_traj[1:, :]):
        dist = torch.pairwise_distance(a, b)
        speed = dist / 0.5
        output_speed.append(speed.view(1, -1))
    output_fake_speed = torch.cat(output_speed, dim=0).unsqueeze(dim=2).permute(1, 0, 2)
    return output_fake_speed


def cal_fse(real_speed, fake_traj):
    last_two_traj_info = fake_traj[-2:, :, :]
    fake_output_speed = fake_speed(last_two_traj_info)
    fse = final_speed_error(real_speed.unsqueeze(dim=2), fake_output_speed)
    return fse


if __name__ == '__main__':
    main()

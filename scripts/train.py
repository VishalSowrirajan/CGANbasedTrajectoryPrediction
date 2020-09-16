import gc
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from csgan.constants import *

from csgan.data.trajectories import data_loader
from csgan.losses import gan_g_loss, gan_d_loss, l2_loss, mean_speed_error, final_speed_error
from csgan.losses import displacement_error, final_displacement_error

from csgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from csgan.losses import relative_to_abs

torch.backends.cudnn.benchmark = True


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def main():
    train_metric = 0
    print("Process Started")
    print("Initializing train dataset")
    train_dset, train_loader = data_loader(TRAIN_DATASET_PATH, train_metric)
    print("Initializing val dataset")
    _, val_loader = data_loader(VAL_DATASET_PATH, train_metric)

    iterations_per_epoch = len(train_dset) / BATCH / D_STEPS
    if NUM_EPOCHS:
        NUM_ITERATIONS = int(iterations_per_epoch * NUM_EPOCHS)

    generator = TrajectoryGenerator()

    generator.apply(init_weights)
    if USE_GPU == 0:
        generator.type(torch.FloatTensor).train()
    else:
        generator.type(torch.cuda.FloatTensor).train()
    print('Here is the generator:')
    print(generator)

    discriminator = TrajectoryDiscriminator()

    discriminator.apply(init_weights)
    if USE_GPU == 0:
        discriminator.type(torch.FloatTensor).train()
    else:
        discriminator.type(torch.cuda.FloatTensor).train()
    print('Here is the discriminator:')
    print(discriminator)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(generator.parameters(), lr=G_LEARNING_RATE)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=D_LEARNING_RATE)

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
    ade_list, fde_list, avg_speed_error = [], [], []
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

                # Maybe save loss
                print('t = {} / {}'.format(t + 1, NUM_ITERATIONS))
                for k, v in sorted(losses_d.items()):
                    print('  [D] {}: {:.3f}'.format(k, v))
                for k, v in sorted(losses_g.items()):
                    print('  [G] {}: {:.3f}'.format(k, v))

                print('Checking stats on val ...')
                metrics_val = check_accuracy(val_loader, generator, discriminator, d_loss_fn)
                print('Checking stats on train ...')
                metrics_train = check_accuracy(train_loader, generator, discriminator, d_loss_fn, limit=True)

                for k, v in sorted(metrics_val.items()):
                    print('  [val] {}: {:.3f}'.format(k, v))
                for k, v in sorted(metrics_train.items()):
                    print('  [train] {}: {:.3f}'.format(k, v))

                ade_list.append(metrics_val['ade'])
                fde_list.append(metrics_val['fde'])
                avg_speed_error.append(metrics_val['msae'])

                if metrics_val.get('ade') == min(ade_list) or metrics_val['ade'] < min(ade_list):
                    print('New low for avg_disp_error')
                if metrics_val.get('fde') == min(fde_list) or metrics_val['fde'] < min(fde_list):
                    print('New low for final_disp_error')
                if metrics_val.get('msae') == min(avg_speed_error) or metrics_val['msae'] < min(avg_speed_error):
                    print('New low for avg_speed_error')

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
    if USE_GPU:
        batch = [tensor.cuda() for tensor in batch]
    else:
        batch = [tensor for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed,
     ped_features) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out, _ = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                              pred_traj_gt, TRAIN_METRIC, SPEED_TO_ADD, ped_features)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
    ped_speed = torch.cat([obs_ped_speed, pred_ped_speed], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, ped_speed, seq_start_end)

    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    optimizer_d.step()

    return losses


def generator_step(batch, generator, discriminator, g_loss_fn, optimizer_g):
    if USE_GPU:
        batch = [tensor.cuda() for tensor in batch]
    else:
        batch = [tensor for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed,
     ped_features) = batch

    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, OBS_LEN:]

    for _ in range(BEST_K):
        generator_out, _ = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj_gt,
                                  TRAIN_METRIC, SPEED_TO_ADD, ped_features)

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

    scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    optimizer_g.step()

    return losses


def check_accuracy(loader, generator, discriminator, d_loss_fn, limit=False):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error = []
    f_disp_error = []
    mean_speed_disp_error = []
    final_speed_disp_error = []
    total_traj = 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            if USE_GPU:
                batch = [tensor.cuda() for tensor in batch]
            else:
                batch = [tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed, ped_features) = batch

            pred_traj_fake_rel, _ = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj_gt,
                                           TRAIN_METRIC, SPEED_TO_ADD, ped_features)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
            loss_mask = loss_mask[:, OBS_LEN:]

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade = cal_ade(pred_traj_gt, pred_traj_fake)
            fde = cal_fde(pred_traj_gt, pred_traj_fake)

            last_pos = obs_traj[-1]
            traj_for_speed_cal = torch.cat([last_pos.unsqueeze(dim=0), pred_traj_fake], dim=0)
            msae = cal_msae(pred_ped_speed, traj_for_speed_cal)
            fse = cal_fse(pred_ped_speed, pred_traj_fake)

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
            ped_speed = torch.cat([obs_ped_speed, pred_ped_speed], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, ped_speed, seq_start_end)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            f_disp_error.append(fde.item())
            mean_speed_disp_error.append(msae.item())
            final_speed_disp_error.append(fse.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            if limit and total_traj >= NUM_SAMPLE_CHECK:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum
    metrics['ade'] = sum(disp_error) / (total_traj * PRED_LEN)
    metrics['fde'] = sum(f_disp_error) / total_traj
    metrics['msae'] = sum(mean_speed_disp_error) / (total_traj * PRED_LEN)
    metrics['fse'] = sum(final_speed_disp_error) / total_traj

    generator.train()
    return metrics


def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel, loss_mask):
    g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode='sum')
    g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum')
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    return ade


def cal_fde(pred_traj_gt, pred_traj_fake):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    return fde


def cal_msae(real_speed, fake_traj):
    fake_output_speed = fake_speed(fake_traj)
    real_speed = real_speed.permute(1, 0, 2)
    msae = mean_speed_error(real_speed, fake_output_speed)
    return msae


def fake_speed(fake_traj):
    output_speed = []
    for a, b in zip(fake_traj[:, :], fake_traj[1:, :]):
        dist = torch.pairwise_distance(a, b)
        speed = dist/0.4
        output_speed.append(speed.view(1, -1))
    output_fake_speed = torch.cat(output_speed, dim=0).unsqueeze(dim=2).permute(1, 0, 2)
    return output_fake_speed


def cal_fse(real_speed, fake_traj):
    last_two_traj_info = fake_traj[-2:, :, :]
    fake_output_speed = fake_speed(last_two_traj_info)
    real_speed = real_speed.permute(1, 0, 2)
    fse = final_speed_error(real_speed, fake_output_speed)
    return fse


if __name__ == '__main__':
    main()

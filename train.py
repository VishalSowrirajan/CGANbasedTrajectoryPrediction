import gc
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from constants import *
from torch.utils.tensorboard import SummaryWriter


from trajectories import data_loader
from utils import gan_g_loss, gan_d_loss, l2_loss, mean_speed_error, \
    final_speed_error, displacement_error, final_displacement_error, relative_to_abs

from models import TrajectoryGenerator, TrajectoryDiscriminator

torch.backends.cudnn.benchmark = True


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def main():
    train_metric = 0
    if MULTI_CONDITIONAL_MODEL and SINGLE_CONDITIONAL_MODEL:
        raise ValueError("Please select either Multi conditional model or single conditional model flag in constants.py")
    print("Process Started")
    if SINGLE_CONDITIONAL_MODEL:
        train_path = SINGLE_TRAIN_DATASET_PATH
        val_path = SINGLE_VAL_DATASET_PATH
    else:
        train_path = MULTI_TRAIN_DATASET_PATH
        val_path = MULTI_VAL_DATASET_PATH
    print("Initializing train dataset")
    train_dset, train_loader = data_loader(train_path, train_metric, 'train')
    print("Initializing val dataset")
    _, val_loader = data_loader(val_path, train_metric, 'val')

    if MULTI_CONDITIONAL_MODEL:
        iterations_per_epoch = len(train_dset) / BATCH_MULTI_CONDITION / D_STEPS
        NUM_ITERATIONS = int(iterations_per_epoch * NUM_EPOCHS_MULTI_CONDITION)
        generator = TrajectoryGenerator(mlp_dim=MLP_INPUT_DIM_MULTI_CONDITION,
                                        h_dim=H_DIM_GENERATOR_MULTI_CONDITION)
        discriminator = TrajectoryDiscriminator(mlp_dim=MLP_INPUT_DIM_MULTI_CONDITION,
                                                h_dim=H_DIM_GENERATOR_SINGLE_CONDITION)
        required_epoch = NUM_EPOCHS_MULTI_CONDITION

    elif SINGLE_CONDITIONAL_MODEL:
        iterations_per_epoch = len(train_dset) / BATCH_SINGLE_CONDITION / D_STEPS

        NUM_ITERATIONS = int(iterations_per_epoch * NUM_EPOCHS_SINGLE_CONDITION)
        generator = TrajectoryGenerator(mlp_dim=MLP_INPUT_DIM_SINGLE_CONDITION,
                                        h_dim=H_DIM_GENERATOR_SINGLE_CONDITION)
        discriminator = TrajectoryDiscriminator(mlp_dim=MLP_INPUT_DIM_SINGLE_CONDITION,
                                                h_dim=H_DIM_GENERATOR_SINGLE_CONDITION)
        required_epoch = NUM_EPOCHS_SINGLE_CONDITION

    print(iterations_per_epoch)
    generator.apply(init_weights)
    generator.type(torch.FloatTensor).train()
    print('Here is the generator:')
    print(generator)

    discriminator.apply(init_weights)
    discriminator.type(torch.FloatTensor).train()
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
    ade_list, fde_list, avg_speed_error, f_speed_error = [], [], [], []

    while epoch < required_epoch:
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
                writer = SummaryWriter()

                ade_list.append(metrics_val['ade'])
                fde_list.append(metrics_val['fde'])
                avg_speed_error.append(metrics_val['msae'])
                f_speed_error.append(metrics_val['fse'])
                writer.add_scalar('val_ade', t, metrics_val['ade'])
                #writer.add_scalar('train_ade', t/100, metrics_train['ade'])
                writer.close()

                if metrics_val.get('ade') == min(ade_list) or metrics_val['ade'] < min(ade_list) or metrics_val.get('fde') == min(fde_list) or metrics_val['fde'] < min(fde_list):
                    checkpoint['g_best_state'] = generator.state_dict()
                if metrics_val.get('ade') == min(ade_list) or metrics_val['ade'] < min(ade_list):
                    print('New low for avg_disp_error')
                if metrics_val.get('fde') == min(fde_list) or metrics_val['fde'] < min(fde_list):
                    print('New low for final_disp_error')

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
    if USE_GPU:
        batch = [tensor.cuda() for tensor in batch]
    else:
        batch = [tensor for tensor in batch]
    if MULTI_CONDITIONAL_MODEL:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed, obs_label, pred_label) = batch
        generator_out = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                  pred_traj_gt, TRAIN_METRIC, obs_label=obs_label, pred_label=pred_label)
    else:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed) = batch
        generator_out = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                  pred_traj_gt, TRAIN_METRIC, obs_label=None, pred_label=None)

    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
    ped_speed = torch.cat([obs_ped_speed, pred_ped_speed], dim=0)
    if MULTI_CONDITIONAL_MODEL:
        label_info = torch.cat([obs_label, pred_label], dim=0)
        scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, label=label_info)
        scores_real = discriminator(traj_real, traj_real_rel, ped_speed, label=label_info)
    else:
        scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, label=None)
        scores_real = discriminator(traj_real, traj_real_rel, ped_speed, label=None)

    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    optimizer_d.step()

    return losses


def generator_step(batch, generator, discriminator, g_loss_fn, optimizer_g):
    """This step is similar to Social GAN Code"""
    if USE_GPU:
        batch = [tensor.cuda() for tensor in batch]
    else:
        batch = [tensor for tensor in batch]
    if MULTI_CONDITIONAL_MODEL:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed,
        obs_label, pred_label) = batch
    else:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed) = batch

    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, OBS_LEN:]

    for _ in range(BEST_K):
        if MULTI_CONDITIONAL_MODEL:
            generator_out = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                  pred_traj_gt, TRAIN_METRIC, obs_label=obs_label, pred_label=pred_label)
        else:
            generator_out = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                      pred_traj_gt, TRAIN_METRIC, obs_label=None, pred_label=None)

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
    if MULTI_CONDITIONAL_MODEL:
        label_info = torch.cat([obs_label, pred_label], dim=0)
        scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, label=label_info)
    else:
        scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, label=None)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
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
    disp_error, f_disp_error, mean_speed_disp_error, final_speed_disp_error = [], [], [], []
    total_traj = 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            if USE_GPU:
                batch = [tensor.cuda() for tensor in batch]
            else:
                batch = [tensor for tensor in batch]
            if MULTI_CONDITIONAL_MODEL:
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed,
                 pred_ped_speed, obs_label, pred_label) = batch
            else:
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed,
                 pred_ped_speed) = batch

            if MULTI_CONDITIONAL_MODEL:
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                  pred_traj_gt, TRAIN_METRIC, obs_label=obs_label, pred_label=pred_label)
            else:
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                      pred_traj_gt, TRAIN_METRIC, obs_label=None, pred_label=None)

            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
            loss_mask = loss_mask[:, OBS_LEN:]

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade = displacement_error(pred_traj_gt, pred_traj_fake)
            fde = final_displacement_error(pred_traj_gt, pred_traj_fake)

            last_pos = obs_traj[-1]
            traj_for_speed_cal = torch.cat([last_pos.unsqueeze(dim=0), pred_traj_fake], dim=0)
            msae = cal_msae(pred_ped_speed, traj_for_speed_cal)
            fse = cal_fse(pred_ped_speed[-1], pred_traj_fake)

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
            ped_speed = torch.cat([obs_ped_speed, pred_ped_speed], dim=0)
            if MULTI_CONDITIONAL_MODEL:
                label_info = torch.cat([obs_label, pred_label], dim=0)
                scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, label=label_info)
                scores_real = discriminator(traj_real, traj_real_rel, ped_speed, label=label_info)
            else:
                scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, label=None)
                scores_real = discriminator(traj_real, traj_real_rel, ped_speed, label=None)

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
            if total_traj >= NUM_SAMPLE_CHECK:
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


def cal_msae(real_speed, fake_traj):
    fake_output_speed = fake_speed(fake_traj)
    real_speed = real_speed.permute(1, 0, 2)
    msae = mean_speed_error(real_speed, fake_output_speed)
    return msae


def fake_speed(fake_traj):
    output_speed = []
    sigmoid_speed = nn.Sigmoid()
    for a, b in zip(fake_traj[:, :], fake_traj[1:, :]):
        dist = torch.pairwise_distance(a, b)
        speed = sigmoid_speed(dist)
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

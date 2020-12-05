import pickle
import torch

from VerifyOutputSpeed import verify_speed
from trajectories import data_loader
from models import TrajectoryGenerator
from utils import displacement_error, final_displacement_error, relative_to_abs, get_dataset_name
from constants import *


def evaluate_helper(error, seq_start_end):
    sum_ = []
    for (start, end) in seq_start_end:
        _error = sum_.append(torch.min(torch.sum(error[start.item():end.item()], dim=0)))
    return sum(sum_)


def evaluate(loader, generator, num_samples):
    ade_outer, fde_outer, simulated_output, total_traj, sequences = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = [tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed,
             obs_label, pred_label) = batch

            ade, fde, sim_op = [], [], []
            total_traj.append(pred_traj_gt.size(1))

            for _ in range(num_samples):
                if TEST_METRIC:
                    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj_gt,
                              TEST_METRIC, SPEED_TO_ADD, obs_label, pred_label)
                else:
                    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed,
                                pred_ped_speed, pred_traj_gt, TEST_METRIC, SPEED_TO_ADD, obs_label, pred_label)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))
                fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))
                sim_op.append(pred_traj_fake)

                for _, (start, end) in enumerate(seq_start_end):
                    num_ped = end - start
                    sequences.append(num_ped)

            ade_outer.append(evaluate_helper(torch.stack(ade, dim=1), seq_start_end))
            fde_outer.append(evaluate_helper(torch.stack(fde, dim=1), seq_start_end))
            simulated_output.append(torch.cat(sim_op, dim=0))

        ade = sum(ade_outer) / (sum(total_traj) * PRED_LEN)
        fde = sum(fde_outer) / (sum(total_traj))
        #simulated_traj_for_visualization = torch.cat(simulated_output, dim=1)
        #sequences = torch.cumsum(torch.stack(sequences, dim=0), dim=0)

        #if TEST_METRIC and VERIFY_OUTPUT_SPEED:
        #    # The speed can be verified for different sequences and this method runs for n number of batches.
        #    verify_speed(simulated_traj_for_visualization, sequences)
        return ade, fde


def main():
    checkpoint = torch.load(CHECKPOINT_NAME)
    generator = TrajectoryGenerator()
    generator.load_state_dict(checkpoint['g_state'])
    generator.train()

    dataset_name = get_dataset_name(TEST_DATASET_PATH)
    _, loader = data_loader(TEST_DATASET_PATH, TEST_METRIC)
    if TEST_METRIC == 1:
        num_samples = 1
    else:
        num_samples = NUM_SAMPLES
    ade, fde = evaluate(loader, generator, num_samples)
    print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(dataset_name, PRED_LEN, ade, fde))


if __name__ == '__main__':
    main()

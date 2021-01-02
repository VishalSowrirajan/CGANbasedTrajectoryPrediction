import pickle
import torch

from VerifyOutputSpeed import verify_speed
from trajectories import data_loader
from models import TrajectoryGenerator
from utils import displacement_error, final_displacement_error, relative_to_abs
from constants import *


def evaluate_helper(error, traj, seq_start_end):
    sum_ = []
    curr_best_traj = []
    for (start, end) in seq_start_end:
        sum_.append(torch.min(torch.sum(error[start.item():end.item()], dim=0)))
        idx = torch.argmin(torch.sum(error[start.item():end.item()], dim=0))
        curr_best_traj.append(traj[idx, :, start:end, :])
    return torch.cat(curr_best_traj, dim=1), sum(sum_)


def evaluate(loader, generator, num_samples):
    ade_outer, fde_outer, simulated_output, total_traj, sequences, labels = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if USE_GPU:
                batch = [tensor.cuda() for tensor in batch]
            else:
                batch = [tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed,
             pred_ped_speed,
             obs_label, pred_label) = batch

            ade, fde, traj_op = [], [], []
            total_traj.append(pred_traj_gt.size(1))
            sequences.append(seq_start_end)
            labels.append(pred_label)

            for _ in range(num_samples):
                if MULTI_CONDITIONAL_MODEL:
                    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                                   pred_traj_gt,
                                                   TEST_METRIC, obs_label=obs_label, pred_label=pred_label)
                else:
                    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                                   pred_traj_gt,
                                                   TEST_METRIC, obs_label=None, pred_label=None)

                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))
                fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))
                traj_op.append(pred_traj_fake.unsqueeze(dim=0))

            best_traj, min_ade_error = evaluate_helper(torch.stack(ade, dim=1), torch.cat(traj_op, dim=0),
                                                       seq_start_end)
            _, min_fde_error = evaluate_helper(torch.stack(fde, dim=1), torch.cat(traj_op, dim=0), seq_start_end)
            ade_outer.append(min_ade_error)
            fde_outer.append(min_fde_error)
            simulated_output.append(best_traj)

        ade = sum(ade_outer) / (sum(total_traj) * PRED_LEN)
        fde = sum(fde_outer) / (sum(total_traj))
        simulated_traj = torch.cat(simulated_output, dim=1)
        all_labels = torch.cat(labels, dim=1)
        last_items_in_sequences = []
        curr_sequences = []
        i = 0
        for sequence_list in sequences:
            last_sequence = sequence_list[-1]
            if i > 0:
                last_items_sum = sum(last_items_in_sequences)
                curr_sequences.append(last_items_sum + sequence_list)
            last_items_in_sequences.append(last_sequence[1])
            if i == 0:
                curr_sequences.append(sequence_list)
                i += 1
                continue

        sequences = torch.cat(curr_sequences, dim=0)

        if TEST_METRIC and VERIFY_OUTPUT_SPEED:
            if SINGLE_CONDITIONAL_MODEL:
                # The speed can be verified for different sequences and this method runs for n number of batches.
                verify_speed(simulated_traj, sequences, labels=None)
            else:
                verify_speed(simulated_traj, sequences, labels=all_labels)

        if ANIMATED_VISUALIZATION_CHECK:
            # Trajectories at User-defined speed for Visualization
            with open('SimulatedTraj.pkl', 'wb') as f:
                pickle.dump(simulated_traj, f, pickle.HIGHEST_PROTOCOL)
            # Sequence list file used for Visualization
            with open('Sequences.pkl', 'wb') as f:
                pickle.dump(sequences, f, pickle.HIGHEST_PROTOCOL)
        return ade, fde


def main():
    checkpoint = torch.load(CHECKPOINT_NAME)
    if MULTI_CONDITIONAL_MODEL:
        generator = TrajectoryGenerator(mlp_dim=MLP_INPUT_DIM_MULTI_CONDITION,
                                        h_dim=H_DIM_GENERATOR_MULTI_CONDITION)
    else:
        generator = TrajectoryGenerator(mlp_dim=MLP_INPUT_DIM_SINGLE_CONDITION,
                                        h_dim=H_DIM_GENERATOR_SINGLE_CONDITION)
    generator.load_state_dict(checkpoint['g_state'])
    if USE_GPU:
        generator.cuda()
    generator.train()

    _, loader = data_loader(TEST_DATASET_PATH, TEST_METRIC)
    ade, fde = evaluate(loader, generator, NUM_SAMPLES)
    print('Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(PRED_LEN, ade, fde))


if __name__ == '__main__':
    main()

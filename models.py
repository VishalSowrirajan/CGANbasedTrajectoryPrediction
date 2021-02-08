import torch
import torch.nn as nn
from constants import *
import math
from utils import relative_to_abs, get_dataset_name


def make_mlp(dim_list, activation='leakyrelu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, h_dim, mlp_input_dim):
        super(Encoder, self).__init__()

        self.mlp_dim = MLP_DIM
        self.h_dim = h_dim
        self.embedding_dim = EMBEDDING_DIM
        self.num_layers = NUM_LAYERS
        self.mlp_input_dim = mlp_input_dim

        self.encoder = nn.LSTM(EMBEDDING_DIM, h_dim, NUM_LAYERS, dropout=DROPOUT)

        self.spatial_embedding = nn.Linear(mlp_input_dim, EMBEDDING_DIM)

    def init_hidden(self, batch):
        if USE_GPU:
            c_s, r_s = torch.zeros(self.num_layers, batch, self.h_dim).cuda(), torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        else:
            c_s, r_s = torch.zeros(self.num_layers, batch, self.h_dim), torch.zeros(self.num_layers, batch, self.h_dim)
        return c_s, r_s

    def forward(self, obs_traj, obs_ped_speed, label=None):
        batch = obs_traj.size(1)
        if MULTI_CONDITIONAL_MODEL:
            embedding_input = torch.cat([obs_traj, obs_ped_speed, label], dim=2)
        else:
            embedding_input = torch.cat([obs_traj, obs_ped_speed], dim=2)
        traj_speed_embedding = self.spatial_embedding(embedding_input.contiguous().view(-1, self.mlp_input_dim))
        obs_traj_embedding = traj_speed_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Decoder(nn.Module):
    def __init__(self, h_dim, mlp_input_dim):
        super(Decoder, self).__init__()

        self.mlp_dim = MLP_DIM
        self.h_dim = h_dim
        self.embedding_dim = EMBEDDING_DIM
        self.mlp_input_dim = mlp_input_dim

        self.decoder = nn.LSTM(EMBEDDING_DIM, h_dim, NUM_LAYERS, dropout=DROPOUT)

        mlp_dims = [h_dim + BOTTLENECK_DIM, MLP_DIM, h_dim]
        self.mlp = make_mlp(mlp_dims, activation=ACTIVATION_RELU, batch_norm=BATCH_NORM, dropout=DROPOUT)
        self.conditionalPoolingModule = ConditionalPoolingModule(h_dim=h_dim, mlp_input_dim=mlp_input_dim)

        self.spatial_embedding = nn.Linear(mlp_input_dim, EMBEDDING_DIM)

        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end, pred_ped_speed, train_or_test, label=None):
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        if train_or_test == 0:
            if MULTI_CONDITIONAL_MODEL:
                last_pos_speed = torch.cat([last_pos_rel, pred_ped_speed[0, :, :], label[0, :, :]], dim=1)
            else:
                last_pos_speed = torch.cat([last_pos_rel, pred_ped_speed[0, :, :]], dim=1)
        else:
            if MULTI_CONDITIONAL_MODEL:
                next_speed = speed_control(pred_ped_speed[0, :, :], seq_start_end, label=label[0, :, :])
                last_pos_speed = torch.cat([last_pos_rel, next_speed, label[0, :, :]], dim=1)
            else:
                next_speed = speed_control(pred_ped_speed[0, :, :], seq_start_end)
                last_pos_speed = torch.cat([last_pos_rel, next_speed], dim=1)
        decoder_input = self.spatial_embedding(last_pos_speed)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            # T0 - T7 is input -- hidden state
        # T8 coordiantes, speed, label
        for id in range(PRED_LEN):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos
            if id + 1 != PRED_LEN:
                if train_or_test == 0:
                    speed = pred_ped_speed[id + 1, :, :]
                    if MULTI_CONDITIONAL_MODEL:
                        curr_label = label[0, :, :]
                else:
                    if SINGLE_CONDITIONAL_MODEL:
                        speed = speed_control(pred_ped_speed[id + 1, :, :], seq_start_end)
                    elif MULTI_CONDITIONAL_MODEL:
                        curr_label = label[0, :, :]
                        speed = speed_control(pred_ped_speed[0, :, :], seq_start_end, label=curr_label)
            if MULTI_CONDITIONAL_MODEL:
                decoder_input = torch.cat([rel_pos, speed, curr_label], dim=1)
            else:
                decoder_input = torch.cat([rel_pos, speed], dim=1)
            decoder_input = self.spatial_embedding(decoder_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)

            if DECODER_TIMESTEP_POOLING:
                if SINGLE_CONDITIONAL_MODEL:
                    pool_h = self.conditionalPoolingModule(state_tuple[0], seq_start_end, train_or_test, curr_pos, speed)
                else:
                    pool_h = self.conditionalPoolingModule(state_tuple[0], seq_start_end, train_or_test, curr_pos,
                                                           speed, label= label[0, :, :])
                decoder_h = torch.cat([state_tuple[0].view(-1, self.h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                state_tuple = (decoder_h.unsqueeze(dim=0), state_tuple[1])

            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel


class ConditionalPoolingModule(nn.Module):
    """The pooling module takes the speed of the pedestrians each other approaching into account"""

    def __init__(self, h_dim, mlp_input_dim):
        super(ConditionalPoolingModule, self).__init__()
        self.mlp_dim = MLP_DIM
        self.h_dim = h_dim
        self.bottleneck_dim = BOTTLENECK_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.mlp_input_dim = mlp_input_dim

        mlp_pre_dim = self.embedding_dim + self.h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, BOTTLENECK_DIM]

        self.pos_embedding = nn.Linear(mlp_input_dim, EMBEDDING_DIM)
        self.mlp_pre_pool = make_mlp(mlp_pre_pool_dims, activation=ACTIVATION_RELU, batch_norm=BATCH_NORM, dropout=DROPOUT)

    def forward(self, h_states, seq_start_end, train_or_test, last_pos, speed, label=None):
        pool_h = []
        if train_or_test == 1:
            speed = speed_control(speed, seq_start_end, label=label)
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden_ped = h_states.view(-1, self.h_dim)[start:end]
            repeat_hstate = curr_hidden_ped.repeat(num_ped, 1).view(num_ped, num_ped, -1)

            if MULTI_CONDITIONAL_MODEL:
                feature = torch.cat([last_pos[start:end], speed[start:end], label[start:end]], dim=1)
            else:
                feature = torch.cat([last_pos[start:end], speed[start:end]], dim=1)
            curr_end_pos_1 = feature.repeat(num_ped, 1)
            curr_end_pos_2 = feature.unsqueeze(dim=1).repeat(1, num_ped, 1).view(-1, self.mlp_input_dim)
            social_features = curr_end_pos_1[:, :2] - curr_end_pos_2[:, :2]
            if MULTI_CONDITIONAL_MODEL:
                social_features_with_speed = torch.cat([social_features, curr_end_pos_1[:, 2].view(-1, 1), curr_end_pos_1[:, -1].view(-1, 1)], dim=1)
            else:
                social_features_with_speed = torch.cat([social_features, curr_end_pos_1[:, 2].view(-1, 1)], dim=1)

            position_feature_embedding = self.pos_embedding(social_features_with_speed.contiguous().view(-1, self.mlp_input_dim))
            pos_mlp_input = torch.cat(
                [repeat_hstate.view(-1, self.h_dim), position_feature_embedding.view(-1, self.embedding_dim)], dim=1)
            pos_attn_h = self.mlp_pre_pool(pos_mlp_input)
            curr_pool_h = pos_attn_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


def speed_control(pred_traj_first_speed, seq_start_end, label=None):
    """This method acts as speed regulator. Using this method, user can add
    speed at one/more frames, stop the pedestrians and so on"""
    for _, (start, end) in enumerate(seq_start_end):
        start = start.item()
        end = end.item()
        if MULTI_CONDITIONAL_MODEL:
            if DIFFERENT_SPEED_MULTI_CONDITION:
                for a, b in zip(range(start, end), label):
                    if torch.eq(b, 0.1):
                        pred_traj_first_speed[a] = sigmoid(AV_SPEED * AV_MAX_SPEED)
                    elif torch.eq(b, 0.2):
                        pred_traj_first_speed[a] = sigmoid(OTHER_SPEED * OTHER_MAX_SPEED)
                    elif torch.eq(b, 0.3):
                        pred_traj_first_speed[a] = sigmoid(AGENT_SPEED * AGENT_MAX_SPEED)
            elif CONSTANT_SPEED_MULTI_CONDITION:
                # To make all pedestrians travel at same and constant speed throughout
                for a, b in zip(range(start, end), label):
                    if torch.eq(b, 0.1):
                        pred_traj_first_speed[a] = sigmoid(CS_MULTI_CONDITION * AV_MAX_SPEED)
                    elif torch.eq(b, 0.2):
                        pred_traj_first_speed[a] = sigmoid(CS_MULTI_CONDITION * OTHER_MAX_SPEED)
                    elif torch.eq(b, 0.3):
                        pred_traj_first_speed[a] = sigmoid(CS_MULTI_CONDITION * AGENT_MAX_SPEED)
        elif SINGLE_CONDITIONAL_MODEL:
            if CONSTANT_SPEED_SINGLE_CONDITION:
                dataset_name = get_dataset_name(SINGLE_TEST_DATASET_PATH)
                if dataset_name == 'eth':
                    speed_to_simulate = ZARA1_MAX_SPEED * CS_SINGLE_CONDITION
                elif dataset_name == 'hotel':
                    speed_to_simulate = ETH_MAX_SPEED * CS_SINGLE_CONDITION
                elif dataset_name == 'univ':
                    speed_to_simulate = ZARA1_MAX_SPEED * CS_SINGLE_CONDITION
                elif dataset_name == 'zara1':
                    speed_to_simulate = ETH_MAX_SPEED * CS_SINGLE_CONDITION
                elif dataset_name == 'zara2':
                    speed_to_simulate = ETH_MAX_SPEED * CS_SINGLE_CONDITION

                # To add an additional speed for each pedestrain and every frame
                for a in range(start, end):
                    pred_traj_first_speed[a] = sigmoid(speed_to_simulate)

            elif STOP_PED_SINGLE_CONDITION:
                # To stop all pedestrians
                for a in range(start, end):
                    pred_traj_first_speed[a] = sigmoid(0)

    return pred_traj_first_speed.view(-1, 1)


class TrajectoryGenerator(nn.Module):
    def __init__(self, mlp_dim, h_dim):
        super(TrajectoryGenerator, self).__init__()

        self.mlp_dim = MLP_DIM
        self.h_dim = h_dim

        self.mlp_input_dim = mlp_dim
        self.h_dim = h_dim

        self.embedding_dim = EMBEDDING_DIM
        self.noise_dim = NOISE_DIM
        self.num_layers = NUM_LAYERS
        self.bottleneck_dim = BOTTLENECK_DIM

        self.encoder = Encoder(h_dim=h_dim, mlp_input_dim=mlp_dim)
        self.decoder = Decoder(h_dim = h_dim, mlp_input_dim=mlp_dim)
        self.conditionalPoolingModule = ConditionalPoolingModule(h_dim=h_dim, mlp_input_dim=mlp_dim)

        self.noise_first_dim = NOISE_DIM[0]

        if POOLING_TYPE:
            mlp_decoder_context_dims = [h_dim + BOTTLENECK_DIM, MLP_DIM, h_dim - self.noise_first_dim]
        else:
            mlp_decoder_context_dims = [h_dim, MLP_DIM, h_dim - self.noise_first_dim]

        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims, activation=ACTIVATION_RELU, batch_norm=BATCH_NORM,
                                            dropout=DROPOUT)

    def add_noise(self, _input, seq_start_end):
        noise_shape = (seq_start_end.size(0),) + self.noise_dim
        if USE_GPU:
            z_decoder = torch.randn(*noise_shape).cuda()
        else:
            z_decoder = torch.randn(*noise_shape)
        _list = []
        for idx, (start, end) in enumerate(seq_start_end):
            noise = z_decoder[idx].view(1, -1).repeat(end.item() - start.item(), 1)
            _list.append(torch.cat([_input[start:end], noise], dim=1))
        decoder_h = torch.cat(_list, dim=0)
        return decoder_h

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj, train_or_test, obs_label=None, pred_label=None, user_noise=None):
        batch = obs_traj_rel.size(1)
        if MULTI_CONDITIONAL_MODEL:
            final_encoder_h = self.encoder(obs_traj_rel, obs_ped_speed, label=obs_label)
        else:
            final_encoder_h = self.encoder(obs_traj_rel, obs_ped_speed, label=None)
        if POOLING_TYPE:
            if train_or_test == 1:
                if MULTI_CONDITIONAL_MODEL:
                    simulated_ped_speed = speed_control(pred_ped_speed[0, :, :], seq_start_end, label=pred_label[0, :, :])
                else:
                    simulated_ped_speed = speed_control(pred_ped_speed[0, :, :], seq_start_end, label=None)
                next_speed = simulated_ped_speed
            else:
                next_speed = pred_ped_speed[0, :, :]
            if MULTI_CONDITIONAL_MODEL:
                sspm = self.conditionalPoolingModule(final_encoder_h, seq_start_end, train_or_test, obs_traj[-1, :, :], next_speed, label=pred_label[0, :, :])
            else:
                sspm = self.conditionalPoolingModule(final_encoder_h, seq_start_end, train_or_test, obs_traj[-1, :, :], next_speed, label=None)
            mlp_decoder_context_input = torch.cat([final_encoder_h.view(-1, self.h_dim), sspm], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(-1, self.h_dim)

        noise_input = self.mlp_decoder_context(mlp_decoder_context_input)

        decoder_h = self.add_noise(noise_input, seq_start_end).unsqueeze(dim=0)
        if USE_GPU:
            decoder_c = torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        else:
            decoder_c = torch.zeros(self.num_layers, batch, self.h_dim)

        state_tuple = (decoder_h, decoder_c)

        if MULTI_CONDITIONAL_MODEL:
            decoder_out = self.decoder(obs_traj[-1], obs_traj_rel[-1], state_tuple, seq_start_end, pred_ped_speed,
            train_or_test, label=pred_label)
        else:
            decoder_out = self.decoder(obs_traj[-1], obs_traj_rel[-1], state_tuple, seq_start_end, pred_ped_speed,
            train_or_test)
        pred_traj_fake_rel = decoder_out

        # LOGGING THE OUTPUT OF ALL SEQUENCES TO TEST THE SPEED AND TRAJECTORIES
        if train_or_test == 1:
            simulated_trajectories = []
            for _, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                obs_test_traj = obs_traj[:, start:end, :]
                pred_test_traj_rel = pred_traj_fake_rel[:, start:end, :]
                #label = pred_label[:, start:end, :]
                pred_test_traj = relative_to_abs(pred_test_traj_rel, obs_test_traj[-1])
                speed_added = pred_ped_speed[0, start:end, :]
                print(speed_added)
                print(pred_test_traj)
                simulated_trajectories.append(pred_test_traj)
        return pred_traj_fake_rel


class TrajectoryDiscriminator(nn.Module):
    def __init__(self, h_dim, mlp_dim):
        super(TrajectoryDiscriminator, self).__init__()

        self.encoder = Encoder(h_dim=h_dim, mlp_input_dim=mlp_dim)

        real_classifier_dims = [h_dim, MLP_DIM, 1]
        self.real_classifier = make_mlp(real_classifier_dims, activation=ACTIVATION_RELU, batch_norm=BATCH_NORM, dropout=DROPOUT)

    def forward(self, traj, traj_rel, ped_speed, label=None):
        if MULTI_CONDITIONAL_MODEL:
            final_h = self.encoder(traj_rel, ped_speed, label=label)  # final layer of the encoder is returned
        else:
            final_h = self.encoder(traj_rel, ped_speed, label=None)  # final layer of the encoder is returned
        scores = self.real_classifier(final_h.squeeze())  # mlp - 64 --> 1024 --> 1
        return scores

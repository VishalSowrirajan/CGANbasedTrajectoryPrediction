import torch
import torch.nn as nn
from constants import *
import math
from utils import relative_to_abs


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
    def __init__(self, h_dim=64):
        super(Encoder, self).__init__()

        self.mlp_dim = MLP_DIM
        self.h_dim = h_dim
        self.embedding_dim = EMBEDDING_DIM
        self.num_layers = NUM_LAYERS

        self.encoder = nn.LSTM(EMBEDDING_DIM, h_dim, NUM_LAYERS, dropout=DROPOUT)

        self.spatial_embedding = nn.Sequential(nn.Linear(4, EMBEDDING_DIM * 2), nn.LeakyReLU(),
                                               nn.Linear(EMBEDDING_DIM * 2, EMBEDDING_DIM))

    def init_hidden(self, batch):
        if USE_GPU == 1:
            c_s, r_s = torch.zeros(self.num_layers, batch, self.h_dim).cuda(), torch.zeros(self.num_layers, batch,
                                                                                           self.h_dim).cuda()
        else:
            c_s, r_s = torch.zeros(self.num_layers, batch, self.h_dim), torch.zeros(self.num_layers, batch, self.h_dim)
        return c_s, r_s

    def forward(self, obs_traj, obs_ped_speed, label_info):
        batch = obs_traj.size(1)
        embedding_input = torch.cat([obs_traj, obs_ped_speed, label_info], dim=2)
        traj_speed_embedding = self.spatial_embedding(embedding_input.contiguous().view(-1, 4))
        obs_traj_embedding = traj_speed_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h.view(batch, -1)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def inverseSigmoid(x):
    return math.log(x / (1 - x))


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.mlp_dim = MLP_DIM
        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM

        self.decoder = nn.LSTM(EMBEDDING_DIM, H_DIM, NUM_LAYERS_DECODER, dropout=DROPOUT)

        mlp_dims = [H_DIM + BOTTLENECK_DIM, MLP_DIM, H_DIM]
        self.mlp = make_mlp(mlp_dims, activation=ACTIVATION, batch_norm=BATCH_NORM, dropout=DROPOUT)

        self.spatial_embedding = nn.Sequential(nn.Linear(4, EMBEDDING_DIM * 2), nn.LeakyReLU(),
                                               nn.Linear(EMBEDDING_DIM * 2, EMBEDDING_DIM))
        self.hidden2pos = nn.Linear(H_DIM, 2)
        self.pool_net = SocialSpeedPoolingModule()

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end, speed_to_add, pred_ped_speed, train_or_test, pred_label):
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        if train_or_test == 0:
            last_pos_speed = torch.cat([last_pos_rel, pred_ped_speed[0, :, :], pred_label[0, :, :]], dim=1)
        else:
            next_speed = speed_control(pred_ped_speed[0, :, :], 0, seq_start_end)
            last_pos_speed = torch.cat([last_pos_rel, next_speed, pred_label[0, :, :]], dim=1)
        decoder_input = self.spatial_embedding(last_pos_speed)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for id in range(PRED_LEN):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos
            if id + 1 != PRED_LEN:
                if train_or_test == 0:
                    speed = pred_ped_speed[id + 1, :, :]
                else:
                    speed = speed_control(pred_ped_speed[id + 1, :, :], 0, seq_start_end)
            decoder_input = torch.cat([rel_pos, speed, pred_label[id, :, :]], dim=1)
            decoder_input = self.spatial_embedding(decoder_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)

            if DECODER_TIMESTEP_POOLING:
                pool_h = self.pool_net(state_tuple[0], seq_start_end, train_or_test, speed_to_add, curr_pos, speed)  # B, 32
                decoder_h = torch.cat([state_tuple[0].view(-1, self.h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                state_tuple = (decoder_h.unsqueeze(dim=0), state_tuple[1])

            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class SocialSpeedPoolingModule(nn.Module):
    """The pooling module takes the speed of the pedestrians each other approaching into account"""

    def __init__(self):
        super(SocialSpeedPoolingModule, self).__init__()
        self.mlp_dim = MLP_DIM
        self.h_dim = H_DIM
        self.bottleneck_dim = BOTTLENECK_DIM
        self.embedding_dim = EMBEDDING_DIM

        mlp_pre_dim = self.embedding_dim + self.h_dim*2
        mlp_pre_pool_dims = [mlp_pre_dim, 512, BOTTLENECK_DIM]

        self.pos_embedding = nn.Sequential(nn.Linear(4, EMBEDDING_DIM * 2), nn.LeakyReLU(),
                                           nn.Linear(EMBEDDING_DIM * 2, EMBEDDING_DIM))
        self.mlp_pre_pool = make_mlp(mlp_pre_pool_dims, activation='leakyrelu', batch_norm=BATCH_NORM, dropout=DROPOUT)

    def forward(self, h_states, seq_start_end, train_or_test, speed_to_add, encOrDec, last_pos, speed, label,
                ped_features=None):
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden_ped = h_states[start:end]
            repeat_hstate = curr_hidden_ped.repeat(num_ped, 1).view(num_ped, num_ped, -1)

            feature = torch.cat([last_pos[start:end], speed[start:end], label[start:end]], dim=1)
            if train_or_test == 1:
                speed = speed_control(speed, SPEED_TO_ADD, seq_start_end)
                feature = torch.cat([last_pos[start:end], speed[start:end], label[start:end]], dim=1)
            curr_end_pos_1 = feature.repeat(num_ped, 1)
            curr_end_pos_2 = feature.unsqueeze(dim=1).repeat(1, num_ped, 1).view(-1, 4)
            social_features = curr_end_pos_1[:, :2] - curr_end_pos_2[:, :2]
            social_features_with_speed = torch.cat([social_features, curr_end_pos_1[:, 2].view(-1, 1), curr_end_pos_1[:, 3].view(-1, 1)], dim=1)

            # POSITION SPEED Pooling
            position_feature_embedding = self.pos_embedding(social_features_with_speed.contiguous().view(-1, 4))
            pos_mlp_input = torch.cat(
                [repeat_hstate.view(-1, self.h_dim*2), position_feature_embedding.view(-1, self.embedding_dim)], dim=1)
            pos_attn_h = self.mlp_pre_pool(pos_mlp_input)
            curr_pool_h = pos_attn_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


def speed_control(pred_traj_first_speed, speed_to_add, seq_start_end, id=None):
    """This method represents the CONTROL MODULE in the paper. Using this module, user can add
    speed at one/more frames, stop the pedestrians and so on"""
    for _, (start, end) in enumerate(seq_start_end):
        start = start.item()
        end = end.item()

        if ADD_SPEED_EVERY_FRAME or ADD_SPEED_PARTICULAR_FRAME:
            if ETH:
                speed_to_add = ETH_MAX_SPEED * SPEED_TO_ADD
            if HOTEL:
                speed_to_add = HOTEL_MAX_SPEED * SPEED_TO_ADD
            if UNIV:
                speed_to_add = UNIV_MAX_SPEED * SPEED_TO_ADD
            if ZARA1:
                speed_to_add = ZARA1_MAX_SPEED * SPEED_TO_ADD
            if ZARA2:
                speed_to_add = ZARA2_MAX_SPEED * SPEED_TO_ADD

            # To add an additional speed for each pedestrain and every frame
            if ADD_SPEED_EVERY_FRAME:
                for a in range(start, end):
                    current_speed = inverseSigmoid(pred_traj_first_speed[a])
                    if sigmoid(current_speed + speed_to_add) < 1:
                        pred_traj_first_speed[a] = sigmoid(current_speed + speed_to_add)
                    else:
                        pred_traj_first_speed[a] = MAX_SPEED
            elif ADD_SPEED_PARTICULAR_FRAME and len(FRAMES_TO_ADD_SPEED) > 0:
                # Add speed to particular frame for all pedestrian
                sorted_frames = FRAMES_TO_ADD_SPEED.sort()
                for frames in sorted_frames:
                    if id == frames and id != 0:
                        for a in range(start, end):
                            pred_traj_first_speed[a] = pred_traj_first_speed[a] + sigmoid(speed_to_add)
                    else:
                        pred_traj_first_speed[a] = pred_traj_first_speed[a]
        elif STOP_PED:
            # To stop all pedestrians
            speed_to_add = 0
            for a in range(start, end):
                pred_traj_first_speed[a] = sigmoid(speed_to_add)
        elif CONSTANT_SPEED_FOR_ALL_PED:
            # To make all pedestrians travel at same and constant speed throughout
            for a in range(start, end):
                pred_traj_first_speed[a] = sigmoid(CONSTANT_SPEED)

    return pred_traj_first_speed.view(-1, 1)


class TrajectoryGenerator(nn.Module):
    def __init__(self):
        super(TrajectoryGenerator, self).__init__()

        self.mlp_dim = MLP_DIM
        self.h_dim = H_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.noise_dim = NOISE_DIM
        self.num_layers = NUM_LAYERS_DECODER
        self.bottleneck_dim = BOTTLENECK_DIM

        self.encoder = Encoder(h_dim=H_DIM)
        self.decoder = Decoder()
        self.social_speed_pooling = SocialSpeedPoolingModule()

        self.noise_first_dim = NOISE_DIM[0]

        if POOLING_TYPE:
            mlp_decoder_context_dims = [H_DIM*2 + BOTTLENECK_DIM, MLP_DIM, H_DIM - self.noise_first_dim]
        else:
            mlp_decoder_context_dims = [H_DIM*2, MLP_DIM, H_DIM - self.noise_first_dim]

        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims, activation=ACTIVATION, batch_norm=BATCH_NORM,
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

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj, train_or_test,
                speed_to_add, obs_label, pred_label, user_noise=None):
        batch = obs_traj_rel.size(1)
        final_encoder_h = self.encoder(obs_traj_rel, obs_ped_speed, obs_label)
        if POOLING_TYPE:
            if train_or_test == 1:
                simulated_ped_speed = speed_control(pred_ped_speed[0, :, :], SPEED_TO_ADD, seq_start_end)
                next_speed = simulated_ped_speed
            else:
                next_speed = pred_ped_speed[0, :, :]
            sspm = self.social_speed_pooling(final_encoder_h, seq_start_end, train_or_test, speed_to_add,
                                             "encoder", obs_traj[-1, :, :], next_speed, obs_label[-1, :, :], ped_features=pred_label)
            mlp_decoder_context_input = torch.cat([final_encoder_h.view(-1, self.h_dim*2), sspm], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(-1, self.h_dim)

        noise_input = self.mlp_decoder_context(mlp_decoder_context_input)

        decoder_h = self.add_noise(noise_input, seq_start_end).unsqueeze(dim=0)
        if USE_GPU:
            decoder_c = torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        else:
            decoder_c = torch.zeros(self.num_layers, batch, self.h_dim)

        state_tuple = (decoder_h, decoder_c)

        decoder_out = self.decoder(
            obs_traj[-1],
            obs_traj_rel[-1],
            state_tuple,
            seq_start_end,
            speed_to_add,
            pred_ped_speed,
            train_or_test,
            pred_label
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        # LOGGING THE OUTPUT OF ALL SEQUENCES TO TEST THE SPEED AND TRAJECTORIES
        if train_or_test == 1:
            outputs = {}
            for _, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                obs_test_traj = obs_traj[:, start:end, :]
                pred_test_traj_rel = pred_traj_fake_rel[:, start:end, :]
                pred_test_traj = relative_to_abs(pred_test_traj_rel, obs_test_traj[-1])
                speed_added = pred_ped_speed[0, start:end, :]
                seq_tuple = (start, end)
                outputs[seq_tuple] = pred_test_traj
        else:
            outputs = None
        return pred_traj_fake_rel, outputs


class TrajectoryDiscriminator(nn.Module):
    def __init__(self):
        super(TrajectoryDiscriminator, self).__init__()

        self.encoder = Encoder(h_dim=H_DIM_DIS)

        real_classifier_dims = [H_DIM_DIS*2, MLP_DIM, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=ACTIVATION,
            batch_norm=BATCH_NORM,
            dropout=DROPOUT
        )

    def forward(self, traj, traj_rel, ped_speed, label_info, seq_start_end=None):
        final_h = self.encoder(traj_rel, ped_speed, label_info)  # final layer of the encoder is returned
        scores = self.real_classifier(final_h.squeeze())  # mlp - 64 --> 1024 --> 1
        return scores

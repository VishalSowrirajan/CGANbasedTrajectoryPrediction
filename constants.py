# DATASET OPTIONS
OBS_LEN = 8
PRED_LEN = 12
MULTI_TRAIN_DATASET_PATH = 'C:/Users/visha/MasterThesis/Argoverse Sub Sample/train'
MULTI_VAL_DATASET_PATH = 'C:/Users/visha/MasterThesis/Argoverse Sub Sample/val'
MULTI_TEST_DATASET_PATH = 'C:/Users/visha/MasterThesis/Argoverse Sub Sample/test'
CHECKPOINT_NAME = 'Checkpoints/Final_Checkpoints_ETH_UCY/With_PM/eth.pt'

SINGLE_TRAIN_DATASET_PATH = 'single_condition_dataset/eth/train'
SINGLE_VAL_DATASET_PATH = 'single_condition_dataset/eth/val'
SINGLE_TEST_DATASET_PATH = 'single_condition_dataset/eth/test'

# NUMBER OF CONDITION FLAG - activate any one of the following flags
SINGLE_CONDITIONAL_MODEL = True  # For salcingle condition
MULTI_CONDITIONAL_MODEL = False  # For multi condition
MODEL_TYPE = 0  # 0 for prediction and 1 for simulation

# MAX SPEEDS FOR ARGOVERSE AND ETH/UCY DATASETS
# for argoverse
AV_MAX_SPEED = 1.6
OTHER_MAX_SPEED = 2.2
AGENT_MAX_SPEED = 2

# for eth/ucy
ETH_MAX_SPEED = 2.0
HOTEL_MAX_SPEED = 2.0
UNIV_MAX_SPEED = 2.0
ZARA1_MAX_SPEED = 2.0
ZARA2_MAX_SPEED = 2.0

# PYTORCH DATA LOADER OPTIONS
NUM_WORKERS = 4
BATCH_MULTI_CONDITION = 32
BATCH_SINGLE_CONDITION = 16
BATCH_NORM = False
ACTIVATION_RELU = 'relu'
ACTIVATION_LEAKYRELU = 'leakyrelu'
ACTIVATION_SIGMOID = 'sigmoid'

# Time between consecutive frames
FRAMES_PER_SECOND_SINGLE_CONDITION = 0.4
FRAMES_PER_SECOND_MULTI_CONDITION = 0.1
NORMALIZATION_FACTOR = 10

# ENCODER DECODER HIDDEN DIMENSION OPTIONS FOR SINGLE AND MULTI CONDITION
H_DIM_GENERATOR_MULTI_CONDITION = 32
H_DIM_DISCRIMINATOR_MULTI_CONDITION = 64

H_DIM_GENERATOR_SINGLE_CONDITION = 32
H_DIM_DISCRIMINATOR_SINGLE_CONDITION = 64

MLP_INPUT_DIM_MULTI_CONDITION = 4
MLP_INPUT_DIM_SINGLE_CONDITION = 3

# HYPER PARAMETERS OPTIONS
G_LEARNING_RATE, D_LEARNING_RATE = 1e-3, 1e-3
NUM_LAYERS = 1
DROPOUT = 0
NUM_EPOCHS_MULTI_CONDITION = 50
NUM_EPOCHS_SINGLE_CONDITION = 50
CHECKPOINT_EVERY = 100
MLP_DIM = 64
EMBEDDING_DIM = 16
BOTTLENECK_DIM = 32
NOISE_DIM = (8, )

L2_LOSS_WEIGHT = 1

NUM_ITERATIONS = 3200
POOLING_TYPE = True
USE_GPU = 0

# SPEED CONTROL FLAGS
TEST_METRIC = 1  # 0 for ground_truth speed. To simulate trajectories, change the flag to 1. This flag is used during testing and inference phase.
TRAIN_METRIC = 0  # Used for training the model with the ground truth
VERIFY_OUTPUT_SPEED = 1

# ADD_SPEED_EVERY_FRAME, STOP_PED, CONSTANT_SPEED_FOR_ALL_PED, ADD_SPEED_PARTICULAR_FRAME - Only one flag out of the 4 can be activated at once.

# Below flag is set to true if multi condition model on argoverse dataset is set to true.
DIFFERENT_SPEED_MULTI_CONDITION = True
AV_SPEED = 0.7
AGENT_SPEED = 0.5
OTHER_SPEED = 0.3

CONSTANT_SPEED_MULTI_CONDITION = False  # CONSTANT_SPEED flag for multi condition
CS_MULTI_CONDITION = 0.2  # Constant speed multi condition

# Below flag is set to true if single condition model on eth/ucy dataset is set to true.

# Change any one of the below flag to True
STOP_PED_SINGLE_CONDITION = False  # Speed 0 will be imposed if the flag is set to True

CONSTANT_SPEED_SINGLE_CONDITION = True
CS_SINGLE_CONDITION = 0.9  # Constant speed single condition

ANIMATED_VISUALIZATION_CHECK = 0

G_STEPS = 1
D_STEPS = 2
SR_STEPS = 1
BEST_K = 10
PRINT_EVERY = 100
NUM_SAMPLES = 20
NOISE = True
NUM_SAMPLE_CHECK = 100
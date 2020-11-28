# DATASET OPTIONS
OBS_LEN = 6
PRED_LEN = 6
TRAIN_DATASET_PATH = 'Cross_Domain_Dataset/train'
VAL_DATASET_PATH = 'Cross_Domain_Dataset/val'
TEST_DATASET_PATH = 'Cross_Domain_Dataset/test'
CHECKPOINT_NAME = 'Checkpoints/ApolloScape_Dataset/chkpt_200epoch_withDecoderPooling.pt'

# DATASET FLAGS FOR ANALYZING THE MAX SPEEDS.
# As there are very fewer speed above the below mentioned speeds, we consider them as outliers.
PED_MAX_SPEED = 18.26
VEH_MAX_SPEED = 28.67
CYC_MAX_SPEED = 21.98

# PYTORCH DATA LOADER OPTIONS
NUM_WORKERS = 4
BATCH = 32
BATCH_NORM = False
ACTIVATION_RELU = 'relu'
ACTIVATION_LEAKYRELU = 'leakyrelu'

# ENCODER DECODER HIDDEN DIMENSION OPTIONS
H_DIM = 32
H_DIM_DIS = 64

# HYPER PARAMETERS OPTIONS
G_LEARNING_RATE, D_LEARNING_RATE = 1e-3, 1e-3
NUM_LAYERS = 1
DROPOUT = 0
NUM_EPOCHS = 200
CHECKPOINT_EVERY = 50
MLP_DIM = 64
EMBEDDING_DIM = 16
BOTTLENECK_DIM = 32
NOISE_DIM = (8, )
DECODER_TIMESTEP_POOLING = True
L2_LOSS_WEIGHT = 1

NUM_ITERATIONS = 3200
POOLING_TYPE = True
USE_GPU = 0

# SPEED CONTROL FLAGS
TEST_METRIC = 1  # 0 for ground_truth speed. To simulate trajectories, change the flag to 1. This flag is used during testing and inference phase.
TRAIN_METRIC = 0  # Used for training the model with the ground truth

# ADD_SPEED_EVERY_FRAME, STOP_PED, CONSTANT_SPEED_FOR_ALL_PED, ADD_SPEED_PARTICULAR_FRAME - Only one flag out of the 4 can be activated at once.
ADD_SPEED_EVERY_FRAME = False  # SPEED_TO_ADD will be active if ADD_SPEED_EVERY_FRAME is True
SPEED_TO_ADD = 0.1

STOP_PED = True  # Makes the speed value as 0

CONSTANT_SPEED_FOR_ALL_PED = False  # CONSTANT_SPEED flag will be active only if CONSTANT_SPEED_FOR_ALL_PED is True
CONSTANT_SPEED = 0.5

ADD_SPEED_PARTICULAR_FRAME = False  # FRAMES_TO_ADD_SPEED flag will be active only if ADD_SPEED_PARTICULAR_FRAME is True
FRAMES_TO_ADD_SPEED = []  # Provide a value between 0 to length of (predicted traj-1)
MAX_SPEED = 0.9999

G_STEPS = 1
D_STEPS = 2
BEST_K = 10
PRINT_EVERY = 100
NUM_SAMPLES = 20
NUM_SAMPLES_CHECK = 5000
NOISE = True
NUM_SAMPLE_CHECK = 5000

# CROSS DOMAIN LOSS FUNCTION WEIGHTED AVERAGES
VEHICLE_COE = 0.2
PEDESTRIAN_COE = 0.58
BICYCLE_COE = 0.22
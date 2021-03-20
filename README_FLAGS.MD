
#Flag descriptions:

- H_DIM: hidden dimensions used for both encoder and decoder LSTM in G
- EMBEDDING_DIM: embedding dimension
- MLP_DIM: MLP dimension
- NOISE_DIM: Gaussian noise dimension
- G_LEARNING_RATE, D_LEARNING_RATE: Learning rate of G and D
- NUM_LAYERS: Number of LSTM layers
- DROPOUT: Dropout to be used in the model
- NUM_EPOCHS: Number of epochs the model is to be trained. 

# Aggregation Module Flags
- POOLING_TYPE: Flag to activate Pooling module component (CSG-P)
- AGGREGATION_TYPE: Flag to activate hidden_state concatenation module (CSG-C)
- ATTENTION_TYPE: Flag to activate Attention module (CSG-A) 

The following flags are used to analyse the maximum speed w.r.t the dataset. 
In ETH/UCY, as each dataset has different max speed value, we implement a common threshold for max_value across all datasets and consider the speeds above the threshold as max_speeds.
In argoverse dataset, we set the ratio of low_speed:high_speed as 100:1.

- OBS_LEN, PRED_LEN: Observed and predicted length of the trajectories
- CHECKPOINT_NAME: Path of the checkpoint at which the weights needs to be stored.
- BATCH: Batch size
- BATCH_NORM: Batch normalization option
- ACTIVATION: We use leakyrelu activation function.
- G_STEPS, D_STEPS: The number of feed-forward and backward passes for generator and discriminator respectively.
- NUM_SAMPLES: Number of samples to test the model
- NOISE: Noise to be added.


# SPEED CONTROL FLAGS
- TRAIN_METRIC: We utilize the next speed information from the ground truth to train the model. This value always holds 0. 
- TEST_METRIC: To check the model with predicted SR speed, use 1. To simulate the model with user-defined speed, use 2.  
- The following flags are applicable only if the test_metric = 2
    - ADD_SPEED_EVERY_FRAME: This flag adds constant speed to every predicted frame. If this flag is 1, then mention the speed to be added in SPEED_TO_ADD flag.
SPEED_TO_ADD = Speed to be added at each frame. The value should be between 0 and 1
    - STOP_PED = This flag stops all the pedestrians. This implies that a speed value of 0 is imposed on all the agents.
    - CONSTANT_SPEED_FOR_ALL_PED = This flags imposes a constant speed to all pedestrians in the scene. 
    - ADD_SPEED_PARTICULAR_FRAME = False
    FRAMES_TO_ADD_SPEED = []  # Provide a value between 0 to length of (predicted traj-1)
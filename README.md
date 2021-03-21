# **Conditional Generative Adversarial Networks for Speed Control in TrajectorySimulation**

# Architecture Overview
Arch overview:
![OriginalSpeedPlot1](https://github.com/VishalSowrirajan/CGANbasedTrajectoryPrediction/blob/feature/full_network/arch.png)


Single agent Simulation with GT (left) and Maximum speed (0.9):
![OriginalSpeedPlot](https://github.com/VishalSowrirajan/CGANbasedTrajectoryPrediction/blob/master/Plots/Real%20and%20Simulated%20Traj%20-%20Max%20Speed.gif)

Single agent Simulation with GT (left) and Minimum speed (0.0):
![IncreasedSpeedPlot](https://github.com/VishalSowrirajan/CGANbasedTrajectoryPrediction/blob/master/Plots/Real%20vs%20Simulated%20-%20Stop%20ped.gif)

**To reproduce the project, run the following command:**

Initially, clone the repo:
````
git clone https://github.com/VishalSowrirajan/CGANbasedTrajectoryPrediction.git
````

To install all dependencies, run:
````
pip install -r requirements.txt
````

To use the pre-trained model, follow:
- For Single-agent, turn USE_GPU flag to 1 and: 
    - To run the model without aggregation: change the flag AGGREGATION_TYPE = 'None' and use the checkpoints from the folder 'NoAgg'
    - To run the model with PM: change the flag AGGREGATION_TYPE = 'pooling' and use the checkpoints from the folder 'PM'
    - To run the model with Concat: change the flag AGGREGATION_TYPE = 'concat' and use the checkpoints from the folder 'Concat'
    - To run the model with Attention: change the flag AGGREGATION_TYPE = 'attention' and use the checkpoints from the folder 'Attention'
- For Multi-agent, turn USE_GPU flag to 0 and change the flag AGGREGATION_TYPE = 'None' and use the checkpoints from the folder 'MultiAgent'


* For Simulation: change the TEST_METRIC to 2 and select one of the following options in CONSTANTS.py:
    * For Single-agent simulation, turn USE_GPU flag to 1 and:
        * To maintain constant speeds for all pedestrians: Change the flag CONSTANT_SPEED_SINGLE_CONDITION to True and enter a value between 0 and 1 in CS_SINGLE_CONDITION variable
        * To stop all the pedestrians: Change the flag STOP_PED_SINGLE_CONDITION to True
    * For Multi-agent Simulation, turn USE_GPU flag to 0 and:
        * To impose different speeds to different agents: Change the flag DIFFERENT_SPEED_MULTI_CONDITION to True and enter a value between 0 and 1 to AV_SPEED, OTHER_SPEED, AGENT_SPEED. 

* For Prediction: change the TEST_METRIC to 1.

To evaluate the model with the simulated trajectories, run:
````
python evaluate_model.py
````

**Note:** The speed value should be 0 < speed > 1

To train the model from scratch, 

Change the CHECKPOINT_NAME (provide a new filename), change the required aggregation flags and run    
````
python train.py
````
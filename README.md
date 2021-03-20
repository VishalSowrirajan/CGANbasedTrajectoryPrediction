# **Conditional Generative Adversarial Networks for Speed Control in TrajectorySimulation**

##Architecture Overview
![Arch](/arch.pdf)


Pedestrian Simulation with Original Speed and Maximum speed:
![OriginalSpeedPlot](https://github.com/VishalSowrirajan/CGANbasedTrajectoryPrediction/blob/master/Plots/Real%20and%20Simulated%20Traj%20-%20Max%20Speed.gif)

Pedestrian Simulation with Original Speed and No speed (Stop pedestrians):
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

Change the necessary fields in Constants.py and Once changed, run the following command:
````
python train.py
````

To evaluate the model with actual ground_truth trajectory speed, run:
````
python evaluate_model.py
````

To simulate trajectories at different speed, change the TEST_METRIC to 2 and select one of the following options in CONSTANTS.py file.
- To maintain constant speeds for all pedestrians: Change the flag CONSTANT_SPEED_FOR_ALL_PED to True and enter a value between 0 and 1 in CONSTANT_SPEED variable
- To stop all the pedestrians: Change the flag STOP_PED to True
- To increase speed at every frames: Change the flag ADD_SPEED_EVERY_FRAME TO True and enter a value between 0 and 1 in SPEED_TO_ADD variable.
- To add speed to a particular frame: Change the flag ADD_SPEED_PARTICULAR_FRAME to True and enter the

After the necessary changes, run:
````
python evaluate_model.py
````
**Note:** The speed value should be 0 < speed > 1
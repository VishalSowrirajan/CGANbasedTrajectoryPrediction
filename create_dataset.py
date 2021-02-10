import numpy as np

def create_data(trajectories, seq_start_end):
    agent_id = 0
    frame_number = 0
    dataset = []
    # traj shape = 181, 12, 2, seq = (tuple of sequence numbers)
    for (start, end) in seq_start_end:
        num_ped = end - start
        curr_seq_agent = trajectories[start:end, :, :].detach().numpy() # 3, 12, 2
        for agents in curr_seq_agent: # 12, 2 - for each agent
            for agent_location in agents: # 2 --> insert frame number and agent number
                dataset.append([frame_number, agent_id, agent_location[0], agent_location[1]])
                frame_number += 1
            agent_id += 1
            frame_number = 0
        frame_number += trajectories.size(1)

    np_dataset = np.asarray(dataset)
    np.savetxt("0.9zara2.txt", np_dataset, delimiter='\t')


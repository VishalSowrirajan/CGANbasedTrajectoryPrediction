import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.pyplot import figure

df00=pd.DataFrame({'time_frame': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                 'traj': [0.2393, 0.2393, 0.2298, 0.1968, 0.1968, 0.1966, 0.1991, 0.0606, 0.0192, 0.0124, 0.0106, 0.0095, 0.0077, 0.0053, 0.0029, 0.0014, 0.0033, 0.0058]})

df01=pd.DataFrame({'time_frame': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                 'traj': [0.2393, 0.2393, 0.2298, 0.1968, 0.1968, 0.1966, 0.1991, 0.1083, 0.0837, 0.0779, 0.0780, 0.0788, 0.0790, 0.0783, 0.0767, 0.0747, 0.0722, 0.0796]})

df02=pd.DataFrame({'time_frame': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                 'traj': [0.2393, 0.2393, 0.2298, 0.1968, 0.1968, 0.1966, 0.1991, 0.1848, 0.1844, 0.1895, 0.1901, 0.1908, 0.1912, 0.1910, 0.1904, 0.1892, 0.1879, 0.1863]})

df03=pd.DataFrame({'time_frame': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                 'traj': [0.2393, 0.2393, 0.2298, 0.1968, 0.1968, 0.1966, 0.1991, 0.2922, 0.2918, 0.2890, 0.2863, 0.2845, 0.2838, 0.2842, 0.2845, 0.2850, 0.2849, 0.2847]})

df04=pd.DataFrame({'time_frame': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                 'traj': [0.2393, 0.2393, 0.2298, 0.1968, 0.1968, 0.1966, 0.1991, 0.3881, 0.3927, 0.3870, 0.3875, 0.3894, 0.3919, 0.3940, 0.3957, 0.3968, 0.3973, 0.3974]})

df05=pd.DataFrame({'time_frame': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                 'traj': [0.2393, 0.2393, 0.2298, 0.1968, 0.1968, 0.1966, 0.1991, 0.4737, 0.5160, 0.5085, 0.5086, 0.5085, 0.5074, 0.5062, 0.5051, 0.5039, 0.5031, 0.5023]})

df06=pd.DataFrame({'time_frame': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                 'traj': [0.2393, 0.2393, 0.2298, 0.1968, 0.1968, 0.1966, 0.1991, 0.5751, 0.6043, 0.5912, 0.5982, 0.6035, 0.6070, 0.6092, 0.6103, 0.6106, 0.6103, 0.6096]})

df07=pd.DataFrame({'time_frame': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                 'traj': [0.2393, 0.2393, 0.2298, 0.1968, 0.1968, 0.1966, 0.1991, 0.6598, 0.6828, 0.6786, 0.6837, 0.6874, 0.6902, 0.6915, 0.6919, 0.6919, 0.6915, 0.6909]})

df08=pd.DataFrame({'time_frame': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                 'traj': [0.2393, 0.2393, 0.2298, 0.1968, 0.1968, 0.1966, 0.1991, 0.6680, 0.7381, 0.7445, 0.7560, 0.7655, 0.7716, 0.7758, 0.7786, 0.7802, 0.7808, 0.7808]})

df09=pd.DataFrame({'time_frame': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                 'traj': [0.2393, 0.2393, 0.2298, 0.1968, 0.1968, 0.1966, 0.1991, 0.7003, 0.8061, 0.8034, 0.8119, 0.8187, 0.8219, 0.8238, 0.8248, 0.8251, 0.8249, 0.8241]})

# multiple line plot
plt.plot('time_frame', 'traj', data=df00[df00['time_frame'] < 8], color='black', linewidth=2, label='Obs')
plt.plot('time_frame', 'traj', data=df00[df00['time_frame'] > 6], marker='.', color='green', linewidth=2, linestyle = '--', label='0')
plt.plot('time_frame', 'traj', data=df01[df01['time_frame'] > 6], marker='.', color='yellow', linewidth=2, linestyle = '--', label='0.1')
plt.plot('time_frame', 'traj', data=df02[df02['time_frame'] > 6], marker='.', color='orange', linewidth=2, linestyle = '--', label='0.2')
plt.plot('time_frame', 'traj', data=df03[df03['time_frame'] > 6], marker='.', color='red', linewidth=2, linestyle = '--', label='0.3')
plt.plot('time_frame', 'traj', data=df04[df04['time_frame'] > 6], marker='.', color='brown', linewidth=2, linestyle = '--', label='0.4')
plt.plot('time_frame', 'traj', data=df05[df05['time_frame'] > 6], marker='.', color='blue', linewidth=2, linestyle = '--', label='0.5')
plt.plot('time_frame', 'traj', data=df06[df06['time_frame'] > 6], marker='.', color='purple', linewidth=2, linestyle = '--', label='0.6')
plt.plot('time_frame', 'traj', data=df07[df07['time_frame'] > 6], marker='.', color='pink', linewidth=2, linestyle = '--', label='0.7')
plt.plot('time_frame', 'traj', data=df08[df08['time_frame'] > 6], marker='.', color='violet', linewidth=2, linestyle = '--', label='0.8')
plt.plot('time_frame', 'traj', data=df09[df09['time_frame'] > 6], marker='.', color='indigo', linewidth=2, linestyle = '--', label='0.9')

#plt.legend(bbox_to_anchor=(1,1), loc="upper left")
plt.legend(loc="upper left")
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
plt.title('Pedestrian Simulated Speeds')
plt.xlabel('Time_Frame')
plt.ylabel('Speed')
#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.show()
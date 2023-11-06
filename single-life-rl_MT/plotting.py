import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle 

# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
# mpl.rcParams['agg.path.chunksize'] = 0
# mpl.rcParams.update( mpl.rc_params() )
# plt.rcParams.update({'font.size': 18})
#plt.rcParams.update({'text.usetex': True})
# plt.rcParams.update({'text.latex.preamble': 'bold'})
# plt.rc('font', family='serif')

# Window_Close
slrl_path = {}
mtsac_path = {}
env_names = ['sawyer_pick_place',       #0
            'sawyer_window_open',       #1
            'sawyer_window_close',      #2
            'sawyer_drawer_open',       #3
            'sawyer_drawer_close',      #4
            'sawyer_button_press',      #5
            'sawyer_push']              #6
env_idx = 6
novelty = 2

title = " ".join([x.capitalize() for x in env_names[env_idx].split('_')])



object_positions_orig=[[0.2, 0.7 , 0.02],
                  [-0.04, 0.705, 0.16],
                  [0.3, 0.705, 0.16],
                  [0.0, 0.9, 0.0],
                  [0.0, 0.6, 0.02],
                  [0., 0.6, 0.0],
                  [0., 0.8, 0.02]]


goals_orig= [[0.5, 0.8, 0.02],
         [0.1, 0.69, 0.16],
         [0.14, 0.69 , 0.02],
         [0.0, 0.74, 0.05],
         [0.0, 0.885, 0.16],
         [0, 0.7 , 0.02],
         [0.1, 0.8, 0.02],
        ]

env_name =env_names[env_idx]
object_name = env_name.split("_")[1].capitalize()
if object_name == 'Window':
    object_name += ' Arm'




# Window Close
# slrl_path['sawyer_window_close'] = f'/home/ishan05/StanfordEE/Spring2023/CS224R/CS-224R-Group-Project/single-life-rl_MT/exp_local/{env_name}/slrl/{novelty}/seed_42/trajectory_sawyer_window_close.pkl'

# mtsac_path['sawyer_window_close'] = f'/home/ishan05/StanfordEE/Spring2023/CS224R/CS-224R-Group-Project/single-life-rl_MT/exp_local/{env_name}/mtsac/{novelty}/trajectory.pkl'



slrl_path[f'{env_name}'] = f'/home/ishan05/StanfordEE/Spring2023/CS224R/CS-224R-Group-Project/single-life-rl_MT/exp_local/{env_name}/slrl/{novelty}/seed_42/trajectory_{env_name}.pkl'

mtsac_path[f'{env_name}'] = f'/home/ishan05/StanfordEE/Spring2023/CS224R/CS-224R-Group-Project/single-life-rl_MT/exp_local/{env_name}/mtsac/{novelty}/trajectory.pkl'




f = open(slrl_path[env_name],'rb')
data_slrl = pickle.load(f)





f = open(mtsac_path[env_name],'rb')
data_mtsac = pickle.load(f)


gripper_pos_mtsac = data_mtsac['starting_gripper_pos']
gripper_pos_slrl = data_mtsac['starting_gripper_pos']


object_pos = data_mtsac['starting_object_pos']
goal_position = data_mtsac['goal_position']


mtsac_traj = np.array(data_mtsac['observations'])
slrl_traj = np.array(data_slrl['observations'])



print(mtsac_traj.shape, slrl_traj.shape)

exit()

fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
# ax.set_axis_off()
# Plot the trajectory

ax.scatter(slrl_traj[:,0], slrl_traj[:,1], marker='o', s = 20 , color = 'deepskyblue', alpha= 0.1, label = 'SLRL')
ax.scatter(mtsac_traj[:,0], mtsac_traj[:,1], marker='o', s = 20 , color = 'blue', alpha= 0.1, label='MTSAC')


# if object_:
ax.scatter(*object_pos[0:2] , marker='o', color = 'red', s = 150 ,  label=f'{object_name} Position')
print(goal_position)
ax.scatter(*goal_position[0:2] , marker='*', color = 'green' , s = 150,  label='Goal')



# if object_:
ax.scatter(*object_positions_orig[env_idx][0:2], marker="o", color = 'orangered', s = 150, alpha=0.3,label=f'Original {object_name} Position' )

ax.scatter(*goals_orig[env_idx][0:2], marker='*', color = 'palegreen' , s = 150, label= 'Original Goal' )



ax.scatter(*mtsac_traj[0][0:2], marker='^', s = 100 , color = 'orange', alpha= 0.7, label = 'Start')

ax.scatter(*slrl_traj[0][0:2], marker='^', s = 100 , color = 'orange', alpha= 0.7)


# ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1 ), ncol=5)

# Set labels for the axes with fontsize
ax.set_xlabel('X Coordinate of End Effector', fontsize=14)
ax.set_ylabel('Y Coordinate of End Effector', fontsize=14)
# ax.set_zlabel('Z', fontsize=12)
ax.set_xlim(-0.5, 0.5)

ax.set_ylim(0.3, 1)
# Set a title for the plot with fontsize
ax.set_title(f'Robotic Arm Trajectory: {title}', fontsize=16)

# Set fontsize for tick labels
ax.tick_params(axis='both', which='major', labelsize=14)


legend2 = ax.legend(loc='lower left', ncol=1)

ax.add_artist(legend2)

ax.tick_params(axis='both', which='major', labelsize=14)

plt.savefig(f'results/{env_name}_{novelty}trajectory.png')

plt.show()
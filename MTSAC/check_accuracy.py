'''
I have changed the files of "mujoco_env.py", "swayer_xyz_env.py". So make sure to check those files. 

I am surpassing the "MjViewer class" of mujoco_py library and instead using "MjRenderContextOffscreen" and the 
"_get_viewer" function in the mujoco_env class is not being used anymore.

'''



# to import he Environments
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import SawyerPickPlaceEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_window_open_v2 import SawyerWindowOpenEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_close_v2 import SawyerDrawerCloseEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_open_v2 import SawyerDrawerOpenEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_window_close_v2 import SawyerWindowCloseEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_push_v2 import SawyerPushEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_button_press_v2 import SawyerButtonPressEnvV2


# to import the expert polciies for the corresponding envs
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
from metaworld.policies.sawyer_window_open_v2_policy import SawyerWindowOpenV2Policy
from metaworld.policies.sawyer_drawer_close_v2_policy import SawyerDrawerCloseV2Policy
from metaworld.policies.sawyer_drawer_open_v2_policy import SawyerDrawerOpenV2Policy
from metaworld.policies.sawyer_window_close_v2_policy import SawyerWindowCloseV2Policy
from metaworld.policies.sawyer_push_v2_policy import SawyerPushV2Policy
from metaworld.policies.sawyer_button_press_v2_policy import SawyerButtonPressV2Policy


import sys, time,os
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import cv2
import numpy as np
import matplotlib.pyplot as plt

#from network_replay_buff import ReplayBuffer, ActorNetwork, CriticNetwork, ValueNetwork
from encoding_network_replay_buff import ReplayBuffer, ActorNetwork, CriticNetwork, ValueNetwork
from SAC import Agent
from tqdm import tqdm

# sys.path.append("/home/tejas/Documents/Stanford/CS 224R/Final Project/Metaworld/metaworld/envs/mujoco/")

# #from metaworld.envs.mujoco.mujoco_env import mujoco_env
# from mujoco_env import MujocoEnv


import pygame
from pygame.locals import QUIT, KEYDOWN
pygame.init()
screen = pygame.display.set_mode((400, 300))



# env = SawyerPickPlaceEnvV2()
# print("Max path length: ", env.max_path_length)
# sys.exit()

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)



envs = [SawyerPickPlaceEnvV2(), 
        SawyerWindowOpenEnvV2(), 
        SawyerWindowCloseEnvV2(),
        SawyerDrawerOpenEnvV2(),
        SawyerDrawerCloseEnvV2(),
        SawyerButtonPressEnvV2(),
        SawyerPushEnvV2()]


expert_policies = [SawyerPickPlaceV2Policy(),
                   SawyerWindowOpenV2Policy(),
                   SawyerWindowCloseV2Policy(),
                   SawyerDrawerOpenV2Policy(),
                   SawyerDrawerCloseV2Policy(),
                   SawyerButtonPressV2Policy(),
                   SawyerPushV2Policy()
                   ]

# task embeddings using Sine Cosine
d = [[np.sin(k*p) for k in range(1,len(envs)+1)] for p in range(1,len(envs)+1)]
# d = [[int(i == j) for i in range(len(envs))] for j in range(len(envs))]


# to append results in the list
mt_sac_reward_results = []
expert_policy_reward_results = []
mt_sac_bool_results = []
expert_policy_bool_results = []


# for i, env in tqdm(enumerate(envs)):

#     mtsac_log_list_per_env = []
#     expert_log_list_per_env = []
#     mtsac_rewards_per_env = []
#     expert_rewards_per_env = []



end_effector_position = []
object_init_position = []
goal_position = []


for i in range(4,5):
    env = envs[i]
    print("Env loaded is: ", env)
    expert_policy = expert_policies[i]

    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.reset()
    env._freeze_rand_vec = True
    lock_action = False
    random_action = False
    output = env.action_space.high
    print("OUTPUT: ", output)

    agent = Agent(input_dims=(46,), env = env, n_actions = env.action_space.shape[0])

    load_checkpoint = True

    if load_checkpoint:
        agent.load_models()
        #env.render()
    

    observation = env.reset()
    # env.render()
    done = False
    mt_sac_score = 0
    counter = 0

    # to append the object init position
    object_init_position.append(observation[4:7])

    # to append the goal position
    goal_position.append(observation[-3:])

    
    # to choose action from the MT_SAC AGENT
    while not done:
        action = agent.choose_action(np.array(list(observation)+d[i]))

        observation_, reward, done, info = env.step(action)

        # to append the end effector position
        end_effector_position.append(observation_[:3])

        time.sleep(0.1)
        mt_sac_score += reward

        #agent.remember(observation, action, reward, observation_, done)

        if not load_checkpoint:
            agent.learn()
        
        observation = observation_
        counter = counter + 1

        if counter == env.max_path_length - 1:
            break
        
        env.render()




end_effector_position = np.array(end_effector_position)
object_init_position= np.array(object_init_position)
goal_position= np.array(goal_position)


end_effector_position_filename = "end_effector_position_filename.npy"
object_init_position_filename = "object_init_position_filename.npy"
goal_position_filename = "goal_position_filename.npy"

np.save(end_effector_position_filename, end_effector_position)
np.save(object_init_position_filename, object_init_position)
np.save(goal_position_filename, goal_position)

print("The new numpy arrays have been saved......")





    
    # # to log the MT-SAC values
    # mtsac_rewards_per_env.append(mt_sac_score)
    # if done == True:
    #     mtsac_log_list_per_env.append("success")
    # else:
    #     mtsac_log_list_per_env.append("failure")


        
        # # TO TAKE EPXERT ACTIONS AND RECORD THE RETURNS
        # observation = env.reset()
        # done = False
        # expert_score = 0
        # counter = 0


        # # to choose action from the EXPERT AGENT
        # while not done:
        #     action = expert_policies[i].get_action(observation)

        #     observation_, reward, done, info = env.step(action)

        #     time.sleep(0.1)
        #     expert_score += reward

        #     #agent.remember(observation, action, reward, observation_, done)

        #     if not load_checkpoint:
        #         agent.learn()
            
        #     observation = observation_
        #     counter = counter + 1

        #     if counter == env.max_path_length - 1:
        #         break
        
        # # to log the expert values
        # expert_rewards_per_env.append(expert_score)
        # if done == True:
        #     expert_log_list_per_env.append("success")
        # else:
        #     expert_log_list_per_env.append("failure")
    
    
#     # to log the lists of list
#     mt_sac_bool_results.append(mtsac_log_list_per_env)
#     expert_policy_bool_results.append(expert_log_list_per_env)

#     mt_sac_reward_results.append(mtsac_rewards_per_env)
#     expert_policy_reward_results.append(expert_rewards_per_env)





# # to convert and save the numpy arrays
# mt_sac_bool_results = np.array(mt_sac_bool_results)
# expert_policy_bool_results = np.array(expert_policy_bool_results)
# mt_sac_reward_results = np.array(mt_sac_reward_results)
# expert_policy_reward_results = np.array(expert_policy_reward_results)


# mt_sac_bool_results_filename = "mt_sac_bool_results.npy"
# expert_policy_bool_results_filename = "expert_policy_bool_results.npy"
# mt_sac_reward_results_filename = "mt_sac_reward_results.npy"
# expert_policy_reward_results_filename = "expert_policy_reward_results.npy"


# np.save(mt_sac_bool_results_filename, mt_sac_bool_results)
# np.save(expert_policy_bool_results_filename, expert_policy_bool_results)
# np.save(mt_sac_reward_results_filename, mt_sac_reward_results)
# np.save(expert_policy_reward_results_filename, expert_policy_reward_results)

# print('Saved the numpy arrays....')

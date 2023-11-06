'''
I have changed the files of "mujoco_env.py", "swayer_xyz_env.py". So make sure to check those files. 

I am surpassing the "MjViewer class" of mujoco_py library and instead using "MjRenderContextOffscreen" and the 
"_get_viewer" function in the mujoco_env class is not being used anymore.

'''

from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import SawyerPickPlaceEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_window_open_v2 import SawyerWindowOpenEnvV2
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

from network_replay_buff import ReplayBuffer
from SAC import Agent, ActorNetwork, CriticNetwork, ValueNetwork, AgentEmbed

from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
from metaworld.policies.sawyer_window_open_v2_policy import SawyerWindowOpenV2Policy
from metaworld.policies.sawyer_window_close_v2_policy import SawyerWindowCloseV2Policy
from metaworld.policies.sawyer_drawer_open_v2_policy import SawyerDrawerOpenV2Policy
from metaworld.policies.sawyer_drawer_close_v2_policy import SawyerDrawerCloseV2Policy
from metaworld.policies.sawyer_button_press_v2_policy import SawyerButtonPressV2Policy
from metaworld.policies.sawyer_push_v2_policy import SawyerPushV2Policy
from env_loader import make
import argparse
import pdb

from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import SawyerPickPlaceEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import SawyerPickPlaceEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_window_open_v2 import SawyerWindowOpenEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_window_close_v2 import SawyerWindowCloseEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_open_v2 import SawyerDrawerOpenEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_close_v2 import SawyerDrawerCloseEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_button_press_v2 import SawyerButtonPressEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_push_v2 import SawyerPushEnvV2
import pickle

root_dir = '/home/ishan05/StanfordEE/Spring2023/CS224R/CS-224R-Group-Project/single-life-rl_MT/exp_local/'
envs = [SawyerPickPlaceEnvV2, 
            SawyerWindowOpenEnvV2, 
            SawyerWindowCloseEnvV2,
            SawyerDrawerOpenEnvV2,
            SawyerDrawerCloseEnvV2,
            SawyerButtonPressEnvV2, SawyerPushEnvV2]

env_names= ['sawyer_pick_place', 'sawyer_window_open', 'sawyer_window_close', 'sawyer_drawer_open', 'sawyer_drawer_close','sawyer_button_press', 'sawyer_push']

# import the learned embeddings here of load the model here
d = [[np.sin(k*p) for k in range(1,len(envs)+1)] for p in range(1,len(envs)+1)]

env_to_task = {}
name_to_env = {}
for i in range(len(env_names)):
    env_to_task[env_names[i]] = np.array(d[i])
    name_to_env[env_names[i]] = envs[i]


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

novelty = 2
idx = 6
def main():
    # envs = [SawyerPickPlaceEnvV2()]
    policies = [SawyerPickPlaceV2Policy(), 
                SawyerWindowOpenV2Policy(), 
                SawyerWindowCloseV2Policy(), 
                SawyerDrawerOpenV2Policy(), 
                SawyerDrawerCloseV2Policy(), 
                SawyerButtonPressV2Policy(),
                SawyerPushV2Policy() ]
    
    env = envs[idx]()
   
    env_name = env_names[idx]
    env._partially_observable = False
    env._freeze_rand_vec = True
    env._set_task_called = True
    env.reset()
    env.render()
  
    agent = Agent(input_dims=(46,), env = env, n_actions = env.action_space.shape[0])
    agent.memory.import_buffer(["/tmp/sac/mtsac_replay_buffer.json"])
    n_episodes = 1

    score_history = []
    eval_score_history = []
    num_episodes = []
    load_checkpoint = True
    agent.load_models()
    logs = {}
    logs['observations'] = []

    env.render()



    if load_checkpoint:
        agent.load_models()
    
    for i in range(n_episodes):
    
        done = False
        score = 0
        counter = 0
        observation = env.reset()
        end_pos = []
        while not done:



            obs = np.array(list(observation)+d[idx])
            action_agent = agent.choose_action(obs, deter = False)
            # action = policies[idx].get_action(observation)
            
            logs['starting_gripper_pos'] = obs[0:3]
            logs['starting_object_pos'] = obs[4:7]
            logs['goal_position'] = env._target_pos
            # print("Here",env._target_pos)
            
            
            # pdb.set_trace()# logs['goal_position'] = env[36:39]
            # logs['goal_position_final'] = env._get_pos_goal()

            # print(logs)


            action = action_agent
            
            # print( action_agent, action)

            observation_, reward, done, info = env.step(action)

            end_pos.append(observation[0:3])
            # time.sleep(0.1)
            next_obs = np.array(list(observation_)+d[idx])

            agent.remember(obs, action, reward, next_obs, done )
            score += reward
            observation = observation_


            env.render()
            # time.sleep(0.1)
            counter = counter + 1

            if counter > 500:    
                break

            
            score_history.append(score)
            num_episodes.append(i)

        if not os.path.exists(root_dir + env_name  + f'/mtsac/{novelty}/'):
            os.makedirs(root_dir + env_name  + f'/mtsac/{novelty}/')
            
        print('Here', len(end_pos))
        with open(root_dir + env_name  + f'/mtsac/{novelty}/' + 'trajectory.pkl', 'wb') as f:        
                logs['observations'] = end_pos
                pickle.dump(logs, f) 

        
        avg_score = np.mean(np.array(score_history[-100:]), axis = 0)

        print('episode = ', i, ' score = ', score, ' avg_score = ', avg_score)
        print("="*100)
        
        
        exit()

    
    # to save the trained models
    if not load_checkpoint:
        agent.save_models()
    
    # to save the replay buffer
    if not load_checkpoint:
        print("Saving replay buffer....")
        agent.memory.save_buffer()
        print('Replay buffer has been saved....')
    
    
    reward_filename = "per_episode_reward_history.npy"
    eval_reward_filename = "eval_per_episode_reward_history.npy"
    episode_filename = 'episodes.npy'

    score_history = np.array(score_history)
    eval_score_history = np.array(eval_score_history)
    num_episodes = np.array(num_episodes)




main()

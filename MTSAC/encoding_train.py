'''
I have changed the files of "mujoco_env.py", "swayer_xyz_env.py". So make sure to check those files. 

I am surpassing the "MjViewer class" of mujoco_py library and instead using "MjRenderContextOffscreen" and the 
"_get_viewer" function in the mujoco_env class is not being used anymore.

'''




from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import SawyerPickPlaceEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_window_open_v2 import SawyerWindowOpenEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_window_close_v2 import SawyerWindowCloseEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_open_v2 import SawyerDrawerOpenEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_close_v2 import SawyerDrawerCloseEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_button_press_v2 import SawyerButtonPressEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_push_v2 import SawyerPushEnvV2
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

from network_replay_buff import ReplayBuffer, ActorNetwork, CriticNetwork, ValueNetwork
from SAC import Agent

from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
from metaworld.policies.sawyer_window_open_v2_policy import SawyerWindowOpenV2Policy
from metaworld.policies.sawyer_window_close_v2_policy import SawyerWindowCloseV2Policy
from metaworld.policies.sawyer_drawer_open_v2_policy import SawyerDrawerOpenV2Policy
from metaworld.policies.sawyer_drawer_close_v2_policy import SawyerDrawerCloseV2Policy
from metaworld.policies.sawyer_button_press_v2_policy import SawyerButtonPressV2Policy
from metaworld.policies.sawyer_push_v2_policy import SawyerPushV2Policy


#from metaworld.envs.mujoco.mujoco_env import mujoco_env
# from mujoco_env import MujocoEnv


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





def main():

    envs = [SawyerPickPlaceEnvV2(), 
            SawyerWindowOpenEnvV2(), 
            SawyerWindowCloseEnvV2(),
            SawyerDrawerOpenEnvV2(),
            SawyerDrawerCloseEnvV2(),
            SawyerButtonPressEnvV2(),
            SawyerPushEnvV2()]
    policies = [SawyerPickPlaceV2Policy(), 
                SawyerWindowOpenV2Policy(),
                SawyerWindowCloseV2Policy(),
                SawyerDrawerOpenV2Policy(),
                SawyerDrawerCloseV2Policy(),
                SawyerButtonPressV2Policy(),
                SawyerPushV2Policy()]
    for env in envs:
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        env.reset()
        env._freeze_rand_vec = True
    lock_action = False
    random_action = False

    d = [[int(i == j) for i in range(len(envs))] for j in range(len(envs))]


    agent = Agent(input_dims=(46,), env = envs[0], n_actions = envs[0].action_space.shape[0])
    #agent.memory.import_buffer(["data/open_window/ann_open_window_replay_buffer.json",
    #                            "data/pick_place/ann_pick_place_replay_buffer.json"])
    #agent.memory.import_buffer(["tmp/sac/mtsac_replay_buffer.json"])

    n_episodes = 10000

    score_history = []
    eval_score_history = [[0 for k in range(len(envs))]]
    num_episodes = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render()
    
    

    for i in range(n_episodes):
        if i%200 == 0:
            observations = [env.reset() for env in envs]
            done = False
            score = [0 for k in range(len(envs))]
            counter = 0

            while not done:
                for j, observation in enumerate(observations):
                    action = agent.choose_action(list(observation)+d[j])

                    observation_, reward, done, info = envs[j].step(action)

                    score[j] += reward
                    observations[j] = observation_
                
                counter = counter + 1

                if counter == envs[0].max_path_length - 1:
                    break



            eval_score_history.append(score)
            avg_score = np.mean(np.array(score_history[-100:]), axis = 0)

            # if avg_score > best_score:
            #     best_score = avg_score
            #     if not load_checkpoint:
            #         agent.save_models()

            print('Eval: episode = ', i, ' score = ', score, ' avg_score = ', avg_score)
            print("="*100)

            best = [score[k]>eval_score_history[-2][k] for k in range(len(envs))]
            print(best)
            if np.sum(best)>3:
                agent.save_models()

            acc = [score[k]>4000 for k in range(len(envs))]
            if np.sum(acc) == len(envs):
                break

        observations = [env.reset() for env in envs]
        done = False
        score = [0 for k in range(len(envs))]
        counter = 0

        while not done:
            for j, observation in enumerate(observations):
                if np.random.rand() < 0.5*(n_episodes-i)/n_episodes:
                    action = policies[j].get_action(observation)
                else:
                    action = agent.choose_action(list(observation)+d[j])

                observation_, reward, done, info = envs[j].step(action)

                score[j] += reward
                observations[j] = observation_

                agent.remember(np.array(list(observation)+d[j]), action, reward, np.array(list(observation_)+d[j]), done)

            if not load_checkpoint:
                agent.learn()

            
            counter = counter + 1

            if counter == envs[0].max_path_length - 1:
                break


        score_history.append(score)
        num_episodes.append(i)
        avg_score = np.mean(np.array(score_history[-100:]), axis = 0)

        print('episode = ', i, ' score = ', score, ' avg_score = ', avg_score)
        print("="*100)


    
    # to save the trained models
    # if not load_checkpoint:
    #     agent.save_models()
    
    # to save the replay buffer
    if not load_checkpoint:
        print("Saving replay buffer....")
        agent.memory.save_buffer()
        print('Replay buffer has been saved....')
    
    
    # to save the number of episodes and the corresponding episode rewards in np arrays
    reward_filename = "per_episode_reward_history.npy"
    eval_reward_filename = "eval_per_episode_reward_history.npy"
    episode_filename = 'episodes.npy'

    score_history = np.array(score_history)
    eval_score_history = np.array(eval_score_history)
    num_episodes = np.array(num_episodes)

    print("Saving the numpy arrays....")

    np.save(reward_filename, score_history)
    np.save(eval_reward_filename, eval_score_history)
    np.save(episode_filename, num_episodes)

    print("Saved the numpy arrays....")




main()

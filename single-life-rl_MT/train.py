import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "osmesa"
import sys
from pathlib import Path
import env_loader
import hydra
import numpy as np
import torch
import random
import utils
from PIL import Image
import time
import utils 
from dm_env import specs
from logger import Logger
from simple_replay_buffer import SimpleReplayBuffer
from video import TrainVideoRecorder
from agents import SACAgent, Discriminator
from backend.timestep import ExtendedTimeStep
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 12})

from SAC import Agent, AgentEmbed
import pdb
import pickle
torch.backends.cudnn.benchmark = True

# Define the environments used for training the MT SAC 
env_names = ['sawyer_pick_place', 
             'sawyer_window_open', 
             'sawyer_window_close', 
             'sawyer_drawer_open', 
             'sawyer_drawer_close',
             'sawyer_button_press', 
             'sawyer_push'
            ]

# Sine Cosine Embeddings
d = [[np.sin(k*p) for k in range(1,len(env_names)+1)] for p in range(1,len(env_names)+1)]

# One Hot Encoding for the tasks
# d_one_hot = [[int(i == j) for i in range(len(env_names))] for j in range(len(env_names))]

env_to_task = {}

for i in range(len(env_names)):
    env_to_task[env_names[i]] = d[i]


# Initialize the SAC agent for the SLRL setting
def make_agent(obs_spec, action_spec, cfg, env_name):
    
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.feature_dim = obs_spec.shape[0]
    return SACAgent(obs_shape=cfg.obs_shape,
                action_shape=cfg.action_shape,
                device=cfg.device,
                lr=cfg.lr,
                feature_dim=cfg.feature_dim,
                hidden_dim=cfg.hidden_dim,
                critic_target_tau=cfg.critic_target_tau, 
                reward_scale_factor=cfg.reward_scale_factor,
                use_tb=cfg.use_tb,
                from_vision=cfg.from_vision,
                env_name=env_name)
    
# Initialize the discriminator for the SLRL setting
def make_discriminator(obs_spec, action_spec, cfg, env_name, discrim_type, mixup, q_weights, num_discrims):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.feature_dim = obs_spec.shape[0]
    return Discriminator(
            discrim_hidden_size=cfg.discrim_hidden_size,
            obs_shape=cfg.obs_shape,
            action_shape=cfg.action_shape,
            device=cfg.device,
            lr=cfg.lr,
            feature_dim=cfg.feature_dim,
            hidden_dim=cfg.hidden_dim,
            critic_target_tau=cfg.critic_target_tau, 
            reward_scale_factor=cfg.reward_scale_factor,
            use_tb=cfg.use_tb,
            from_vision=cfg.from_vision,
            env_name=env_name,
            discrim_type=discrim_type,
            mixup=mixup,
            q_weights=q_weights,
            num_discrims=num_discrims,)


# The workspace is used for the SLRL training and logging the whole experiment
class Workspace:
    def __init__(self, cfg, orig_dir):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.orig_dir = orig_dir

        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)


        self.obs_spec = specs.Array((46,), np.float32, 'observation')
        
        self.setup()
        
        if cfg.task_encoding == 'trainable':
            self.agent = AgentEmbed(env = self.train_env)
        else:
            self.agent = Agent(env = self.train_env)

        # Change this to observation + taskid
        # Discriminator type is state or state-action pair
        if self.cfg.use_discrim:
            self.discriminator = make_discriminator(self.obs_spec,
                                                  self.train_env.action_spec(),
                                                  self.cfg.agent,
                                                  self.cfg.env_name,
                                                     discrim_type=self.cfg.discrim_type,
                                                     mixup=self.cfg.mixup,
                                                     q_weights=self.cfg.q_weights,
                                                      num_discrims=self.cfg.num_discrims)
        
        self.timer = utils.Timer()
        self.timer._start_time = time.time()
        self._global_step = -self.cfg.num_pretraining_steps
        print("Global step", self._global_step)
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.train_env, self.eval_env, self.reset_states, self.goal_states, self.forward_demos = env_loader.make(self.cfg.env_name, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.resets, orig_dir=self.orig_dir)
        
        # Not used as the SAC agent is trained using different code
        if self.cfg.resets:
            _, self.train_env, self.reset_states, self.goal_states, self.forward_demos = env_loader.make(self.cfg.env_name, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.resets, orig_dir=self.orig_dir)
    
        # Observation space plus taskid
        data_specs = ( self.obs_spec,
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))
    
        self.replay_storage_f = SimpleReplayBuffer(data_specs,
                                                       self.cfg.replay_buffer_size,
                                                       self.cfg.batch_size,
                                                       self.work_dir / 'forward_buffer',
                                                  self.cfg.discount)
    
        self.online_buffer = SimpleReplayBuffer(data_specs,
                                                       self.cfg.replay_buffer_size,
                                                       self.cfg.batch_size,
                                                       self.work_dir / 'forward_buffer',
                                               self.cfg.discount,
                                               time_step=self.cfg.time_step)
        self.prior_buffers = []
        
        # We have a single discriminator but still have kept the for loop to allow for more if needed 
        for _ in range(self.cfg.num_discrims):
            self.prior_buffers.append(SimpleReplayBuffer(data_specs,
                                                       self.cfg.prior_buffer_size, 
                                                       self.cfg.batch_size,
                                                       self.work_dir / 'forward_buffer',
                                                         self.cfg.discount,
                                                         time_step=self.cfg.time_step,
                                                       q_weights=self.cfg.q_weights,
                                                       rl_pretraining=self.cfg.rl_pretraining))
        
        self._forward_iter = None 

        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None, self.cfg.env_name)


    @property
    def forward_iter(self):
        if self._forward_iter is None:
            self._forward_iter = iter(self.replay_storage_f)
        return self._forward_iter
    
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat
    
    def save_im(self, im, name):
        img = Image.fromarray(im.astype(np.uint8)) 
        img.save(name)

    def save_gif(self, ims, name):
        imageio.mimsave(name, ims, fps=len(ims)/100)
    
        

    def train(self, snapshot_dir=None):
        train_until_step = utils.Until(self.cfg.online_steps, 
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_init_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        if self.cfg.rl_pretraining:
            time_step = self.eval_env.reset()
            _, self.eval_env_pretraining, _, _, _ = env_loader.make(self.cfg.env_name, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.resets, orig_dir=self.orig_dir)
        else:
            self.train_env._set_task_called = True
            time_step = self.train_env.reset()
        dummy_action = time_step.action

        if self.forward_demos and (not self.cfg.rl_pretraining or self.cfg.use_demos):
            self.replay_storage_f.add_offline_data(self.forward_demos, dummy_action, env_name=self.cfg.env_name)
            
            for buffer in self.prior_buffers:
                _ = buffer.add_offline_data(self.forward_demos, dummy_action, env_name=self.cfg.env_name)
        
        prior_iters = []
        
        for d in range(self.cfg.num_discrims):
            prior_iters.append(iter(self.prior_buffers[d])) 
     
        online_iter = iter(self.online_buffer)
        
        cur_agent = self.agent
        cur_buffer = self.replay_storage_f
        
        cur_iter = self.forward_iter

        if self.cfg.save_train_video:
            self.train_video_recorder.init(self.train_env)
    
        metrics = None
        episode_step, episode_reward = 0, 0
        past_timesteps = []
        online_rews = [] # all online rewards
        online_qvals = [] # all online qvals
        end_effector = [] # record the arm position

        # To log the end effector, starting positions and the whole trajectory of the arm
        logs = {}        
        cur_reward = torch.tensor(0.0).cuda()
        counter = 0
        while train_until_step(self.global_step):
            
            '''Start single episode'''
            if self.global_step == 0:
                print("Starting single episode")
                # Reset the training environment to ensure that the goal and the object positions are set according to our novelties
                time_step = self.train_env.reset()

                logs['starting_gripper_pos'] = time_step.observation[0:3]
                logs['starting_object_pos'] = time_step.observation[4:7]
                logs['goal_position'] = self.train_env.gym_env._target_pos
                logs['goal_position_final'] = self.train_env.gym_env._get_pos_goal()
                # print(self.train_env.gym_env._target_pos)
                # logs['goal_position'] = time_step.observation[36:39]
                logs['observation'] = []

                # Append the task ID to the current observation using the dictionary specified after the imports
                # 
                #  
                time_step =  ExtendedTimeStep(
                                                observation=np.concatenate( (time_step.observation, env_to_task[self.cfg.env_name]), axis = 0 ),       step_type=time_step.step_type,
                                                action=time_step.action,
                                                reward=time_step.reward,
                                                discount=time_step.discount
                                            )


                cur_buffer.add(time_step)

                # Basically used to pretrain the algorithm used to the environment for few hundred step initially. In out approach we train the MT-SAC online using the prior data and the online buffer udpated as the agent takes actions                
                if self.cfg.rl_pretraining:
                    min_q, max_q = cur_buffer.load_buffer(f'{snapshot_dir}/', 
                                                          self.cfg.prior_buffer_size, 
                                                          self.cfg.env_name, 
                                                          agent = self.agent,
                                                          taskid = env_to_task[self.cfg.env_name])
                    if self.prior_buffers[0].__len__() == 0:
                        _, _ = self.prior_buffers[0].load_buffer(f'{snapshot_dir}/', self.cfg.prior_buffer_size, agent = self.agent, taskid=env_to_task[self.cfg.env_name])
                    
            '''Logging and eval'''
            criteria = self.global_step % 500 == 0 
            # Since the metaworld uses a step size of 500, update the global step as the agent takes 500 steps in the online setting
            if criteria: 
               
                # Always update the logs and save the experiment data
                with open(str(self.work_dir / 'trajectory_') + self.cfg.env_name + '.pkl', 'wb') as f:
                    
                    logs['observations'] = end_effector
                    pickle.dump(logs, f) 
                    
                if metrics is not None:
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('forward_buffer_size', len(self.replay_storage_f))
                        log('step', self.global_step)
                        # log('cur_reward', cur_reward.detach().item())
                
                
                if self.global_step % 500 == 0 :
                    episode_step, episode_reward = 0, 0
                self._global_episode += 1
                
             
            # The main part of the single life RL online learning setting
            if self.global_step >= 0 or self.cfg.rl_pretraining:
                '''Sample action'''
                with torch.no_grad():
                        # From the above timestep intialization, the observation has the correct shape. Pass directly to the MT-SAC model
                        obs=time_step.observation

                        # Goal Masking for the SLRL Ablation Study
                        obs[36:39] = np.array([0,0,0])
                        # The agent return a sample action using the output mean and the standard deviation for calculating a stochastic action output.
                        # Should take the mean action during the evaluation phase technically
                        action = cur_agent.choose_action(obs)

                # Update the end effector position in our logs
                end_effector.append(obs[0:3])
                # print('Step')
                time_step = self.train_env.step(action)
                self.train_env.render()

                time_step =  ExtendedTimeStep(
                                                observation=np.concatenate( (time_step.observation, env_to_task[self.cfg.env_name]), axis = 0),       step_type=time_step.step_type,
                                                action=time_step.action,
                                                reward=time_step.reward,
                                                discount=time_step.discount
                                            )
                # This is the original reward computed by the metaworld environment after taking the step in the environment
                orig_reward = time_step.reward

                # This is the total reward for the entire single life experiment
                online_rews.append(cur_reward.detach().item())
                
                # Compute the Q-value for the observation using the critic target model to understand the qaulity/userfulness of the current position of the arm
                with torch.no_grad():
                    Q1 = self.agent.critic_2(torch.FloatTensor(time_step.observation).view(1,-1).cuda(), torch.FloatTensor(time_step.action).view(1,-1).cuda())
                online_qvals.append(Q1.detach().item())
               
                # Check if the agent completes the task
                success_criteria = (time_step.step_type == 2)
                if success_criteria or self.global_step == self.cfg.online_steps - 1:
                    
                    time_step = ExtendedTimeStep(observation=time_step.observation,
                                                 step_type=2,
                                                 action=action,
                                                 reward=time_step.reward,
                                                 discount=time_step.discount)
                    print("Completed task in steps", self.global_step, time_step)

                    with open(str(self.work_dir / 'trajectory_') + self.cfg.env_name + '.pkl', 'wb') as f:                        
                        logs['observations'] = end_effector
                        pickle.dump(logs, f) 

                    with open(f"{self.work_dir}/total_steps.txt", 'w') as f:
                        f.write(str(self.global_step))
                    
                    print('Total Steps', len(end_effector))
                    exit()
                    
                # Total episode reward
                episode_reward += orig_reward                
                # Add to buffer
                cur_buffer.add(time_step)
                self.online_buffer.add(time_step)
                episode_step += 1
                
           ##############################################################################################    
            if self.cfg.use_discrim:
                if self.global_step % self.cfg.discriminator.train_interval == 0 and self.online_buffer.__len__() > self.cfg.discriminator.batch_size:
                    
                    for k in range(self.cfg.discriminator.train_steps_per_iteration):
                        metrics = self.discriminator.update_discriminators( pos_replay_iter = prior_iters, 
                                                                            neg_replay_iter = online_iter, 
                                                                            val_function = self.agent.critic_1, 
                                                                            current_val = cur_reward, 
                                                                            current_obs = time_step, 
                                                                            min_q = min_q, 
                                                                            max_q = max_q, 
                                                                            task_id = env_to_task[self.cfg.env_name],
                                                                            baseline = self.cfg.baseline)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

            if not seed_until_step(self.global_step):

                if self.cfg.use_discrim and self.online_buffer.__len__() > self.cfg.discriminator.batch_size:
                    trans_tuple, original_reward = self.discriminator.transition_tuple(cur_iter)
                    metrics = cur_agent.learn(trans_tuple, self.global_step)
                    metrics['original_reward'] = original_reward.mean()

                    if len(past_timesteps) > 10: # for logging
                        del past_timesteps[0]
                    past_timesteps.append(time_step)
                    old_time_step = past_timesteps[0]
                    latest_tuple, original_reward = self.discriminator.transition_tuple(cur_iter, cur_time_step=time_step, old_time_step=old_time_step)
                    _, _, latest_reward = latest_tuple
                    actual_reward, disc_s = latest_reward
                    metrics['latest_r'] = actual_reward
                    metrics['disc_s'] = disc_s
                    cur_reward = disc_s # Use latest discriminator score as baseline val
                else:
                    obs, action, reward, discount, next_obs, step_type, next_step_type, idxs, q_vals  = utils.to_torch(next(cur_iter), 'cuda:0')
                    trans_tuple = (obs, action, reward, discount, next_obs, step_type, next_step_type)
                    
                    metrics = cur_agent.learn(trans_tuple, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')            
           
            self._global_step += 1    
           


        with open(str(self.work_dir / 'trajectory_') + self.cfg.env_name + '.pkl', 'wb') as f:
            pickle.dump(np.array(end_effector), f) 

    def save_snapshot(self, epoch=None):
        snapshot = self.work_dir / 'snapshot.pt'
        if epoch: snapshot = self.work_dir / f'snapshot{epoch}.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, dirname=None):
        if dirname: 
            payload = torch.load(dirname)
        else: 
            snapshot = self.work_dir / 'snapshot.pt'
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
        self._global_step = -self.cfg.num_pretraining_steps

# Add the environment of Sawyer along with the original dataset
@hydra.main(config_path='./', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd
    print(cfg)
    orig_dir = hydra.utils.get_original_cwd()
    workspace = W(cfg, orig_dir)
    snapshot_dir = None

    path = "/tmp/sac/mtsac_replay_buffer.json"
    workspace.agent.memory.import_buffer([path])
    workspace.agent.load_models()
   
    workspace.train(snapshot_dir)


if __name__ == '__main__':
    main()
    

    

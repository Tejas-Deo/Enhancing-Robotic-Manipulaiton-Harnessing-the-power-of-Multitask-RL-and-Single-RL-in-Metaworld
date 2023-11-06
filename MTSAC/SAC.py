import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from network_replay_buff import ReplayBuffer, ActorNetwork, CriticNetwork, ValueNetwork




class Agent():

    '''
    reward_scale - Reward scaling is an imp hyperparameter to consider as it drives the entropy
    Tau - factor by which we are going to modulate the factors of our target value network (there is a value
    network and target value network. So instead of a hard copy, we are doing a soft copy by detuning the 
    parameters)

    need to change the actions and the reward scale
    '''
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[46],
            env=None, gamma=0.99, n_actions=4, max_size=200000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                    name='actor', max_action=env.action_space.high)
        
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_1')
        
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_2')
        
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    
    def choose_action(self, observation):
        state = torch.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    
    '''
    to soft update the parameters of the target value network wrt to the value network
    '''
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    
    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()


    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint("value_base")
        self.target_value.load_checkpoint("value_target")
        self.critic_1.load_checkpoint("critic_1")
        self.critic_2.load_checkpoint("critic_2")


    def learn(self):
        # go back to the program if we do not have sufficient transitions i.e. sufficient data in the replay buffer
        if self.memory.mem_cntr < self.batch_size:
            return

        # print("Collected enough data and have started learning...")
        # print("="*50)
        
        # to sample our buffer
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # to transform numpy arrays to pytorch tensors
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)  # these are the actions sampled fro the replay buffer

        # to calculate the value current state and next state via the value and target value networks
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        # to set the terminal states value to be 0
        value_[done] = 0.0

        # to get the actions according to the new policy wihtout using the reparameterization trick
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)

        # to critic values under the new policy. NOTE: Using "actions" and not "action"
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)  # to overcome overestimation
        critic_value = critic_value.view(-1)

        # to define the VALUE NETWORK LOSS
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        
        # to define the ACTOR NETWORK loss
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        
        # to define the CRITIC NETWORK LOSS
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        # to add the entropy term so as to encourage exploration
        q_hat = self.scale*reward + self.gamma*value_

        # NOTE: using the action from the replay buffer
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # to update the target value network parameters
        self.update_network_parameters()




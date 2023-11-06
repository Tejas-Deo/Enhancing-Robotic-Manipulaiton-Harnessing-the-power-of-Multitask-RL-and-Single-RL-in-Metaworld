import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import json


'''
To define the class for Replay Buffer
'''
class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        '''
        max_size = maximum size of the replay buffer
        input_shape = observation space
        n_actions = # actinos the agent can take

        to store everything in individual arrays
        '''
        self.mem_size = max_size
        self.mem_cntr = 0

        # to store the states 
        self.state_memory = np.zeros((self.mem_size, *input_shape))

        # to store the states that we get after taking an action
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))

        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)

        # to store the "done" commands (the terminal flags)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    
    def store_transition(self, state, action, reward, state_, done):
        '''
        To store the transitions, "state_" stands for the next state
        '''
        index = self.mem_cntr % self.mem_size

        #print("Inside store_transition function...")

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    
    def sample_buffer(self, batch_size):
        '''
        To sample from the buffer
        '''
        max_mem = min(self.mem_cntr, self.mem_size)

        # to sample
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
    
    def import_buffer(self, files):
        # Import replay buffer from json file

        state_memory = []
        new_state_memory = []
        action_memory = []
        reward_memory = []
        terminal_memory = []

        for file in files:
            with open(file, "r") as f:
                json_string = f.read()
                data = json.loads(json_string)

            state_memory += data["self.state_memory"]
            new_state_memory += data["self.new_state_memory"]
            action_memory += data["self.action_memory"]
            reward_memory += data["self.reward_memory"]
            terminal_memory += data["self.terminal_memory"]

        # to store the states 
        self.state_memory = np.array(state_memory)

        # to store the states that we get after taking an action
        self.new_state_memory = np.array(new_state_memory)

        self.action_memory = np.array(action_memory)
        self.reward_memory = np.array(reward_memory)

        # to store the "done" commands (the terminal flags)
        self.terminal_memory = np.array(terminal_memory)
            
    def save_buffer(self):

        # Create a dictionary to store the numpy arrays with their labels
        arrays_dict = {
            "self.state_memory": self.state_memory.tolist(),
            "self.new_state_memory": self.new_state_memory.tolist(),
            "self.action_memory": self.action_memory.tolist(),
            "self.reward_memory": self.reward_memory.tolist(),
            "self.terminal_memory": self.terminal_memory.tolist()
        }

        # Convert the dictionary to a JSON string
        json_string = json.dumps(arrays_dict)

        # Save the JSON string to a file
        with open("pick_place_replay_buffer.json", "w") as f:
            f.write(json_string)    




''' 
To define the critic network

Outputs the value of Q(s,a); same as we did in the assignment
'''
class CriticNetwork(nn.Module):

    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='tmp/sac'):
         
        '''
        beta - it is the learning rate
        input dims - # input dimensions (obs)
        '''
        
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # directly passing the state and action pair together
        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # output layer - outputting the value of Q(s,a) and so in a scalar value
        self.q = nn.Linear(self.fc2_dims, 1) 

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # to transfer everything to GPU
        self.to(self.device)

    
    def forward(self, state, action):
        # along the batch dimension
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    
    def load_checkpoint(self, name):

        if name == "critic_1":
            critic_checkpoint_path = r"/home/tejas/Documents/Stanford/CS 224R/Final Project/Metaworld/tmp/sac/critic_1_sac.zip"
            print("Critic 1 loaded!")

        if name == "critic_2":
            critic_checkpoint_path = r"/home/tejas/Documents/Stanford/CS 224R/Final Project/Metaworld/tmp/sac/critic_2_sac.zip"
            print("Critic 2 loaded!!")

        #self.load_state_dict(torch.load(self.checkpoint_file))
        self.load_state_dict(torch.load(critic_checkpoint_path))



''''
To define the value network. 

To estimate the value of being in a particular state 
'''
class ValueNetwork(nn.Module):

    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
            name='value', chkpt_dir='tmp/sac'):
        
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)

        # outputting a scalar quantity
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    
    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    
    def load_checkpoint(self, name):
        if name == "value_target":
            value_load_checkpoint = r"/home/tejas/Documents/Stanford/CS 224R/Final Project/Metaworld/tmp/sac/target_value_sac.zip"
            print("Target Value network loaded!")
        
        if name == "value_base":
            value_load_checkpoint = r"/home/tejas/Documents/Stanford/CS 224R/Final Project/Metaworld/tmp/sac/value_sac.zip"
            print("Value network loaded!!")

        #self.load_state_dict(torch.load(self.checkpoint_file))
        self.load_state_dict(torch.load(value_load_checkpoint))




'''
To define the SAC

Outputting a Normal distribution with mean and std
'''
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, 
            fc2_dims=256, n_actions=4, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        '''
        this is the scaling factor for the action space as we have tanh as the output. but the range of actions in the metaworld is btw
        -1 and 1 so we are not using this max_action variable or setting it to 1
        '''
        self.max_action = max_action

        # to avoid log or division by 0
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # mean of the distribution for the policy; equal to the number of actions that we can take
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        # to define the standard deviation
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    
    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        # to predict the mean and std using the same inpui.e. output of fc2
        mu = self.mu(prob) 
        sigma = self.sigma(prob)

        # to prevent the distribution from getting too big
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        # to sample the actions from the distribution
        if reparameterize:
            # if we want to add more noise into the sampled action
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        #action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)
        # this is to take the action in the env 
        action = torch.tanh(actions).to(self.device)

        # to calculate the loss wrt to the sampled action
        log_probs = probabilities.log_prob(actions)

        # to avoid dividing by 0
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    
    def load_checkpoint(self):
        actor_load_checkpoint = r"/home/tejas/Documents/Stanford/CS 224R/Final Project/Metaworld/tmp/sac/actor_sac.zip"
        #self.load_state_dict(torch.load(self.checkpoint_file))
        self.load_state_dict(torch.load(actor_load_checkpoint))
        print("Actor model loaded!")















import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

import agent

from datetime import datetime

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class PrioritizingAgent(agent.Agent):
    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)
        
        self.buffer = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory

        td_error = self.td_error(state, action, reward, next_state, done).item()
        
        if torch.is_tensor(state):
            state = np.array(state.cpu())
        if torch.is_tensor(action):
            action = np.array(action.cpu())
        if torch.is_tensor(reward):
            reward = np.array(reward.cpu())
        if torch.is_tensor(next_state):
            next_state = np.array(next_state.cpu())
        if torch.is_tensor(done):
            done = np.array(done.cpu())
            
        #td_error = 0
        self.buffer.add(state, action, reward, next_state, done, td_error)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.buffer) > BATCH_SIZE:
                experiences = self.buffer.sample()
                self.learn(experiences, GAMMA)
    
    def td_error(self, state, action, reward, next_state, done):
        
        state_cp = torch.from_numpy(state).float().to(device) if type(state) is np.ndarray else state.clone()
        
        next_state_cp = torch.from_numpy(next_state).float().to(device) if type(next_state) is np.ndarray else next_state.clone()
                
        argmax_index = self.qnetwork_target(next_state_cp).detach()

        argmax_index = argmax_index.argmax(0).item() 
        
        td_error = reward + self.gamma*self.qnetwork_local(next_state_cp).detach()[argmax_index]*(1 - done) - self.qnetwork_target(state_cp).detach()[action]
        
        return np.abs(td_error.cpu().numpy())
    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done, indexes) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indexes = experiences
                
        #Double Q-Learning using the Target weights to select the max action, and the local to evaluate it
        
        next_states_actions = self.qnetwork_target(next_states).detach()       
        argmax_index = next_states_actions.argmax(1)
        td_target = rewards + gamma*self.qnetwork_local(next_states).detach().gather(1, argmax_index.unsqueeze(1))*(1 - dones)

        #td_target = rewards + gamma*self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)*(1 - dones)        
        outputs = self.qnetwork_local(states)
        
        for i in range(len(states)):
            state, action, reward, next_state, done, index = states[i], actions[i], rewards[i], next_states[i], dones[i], indexes[i]
            
            td_error = td_target[i] - outputs[i][action]
            
            if torch.is_tensor(state):
                state = np.array(state.cpu())
            if torch.is_tensor(action):
                action = np.array(action.cpu())
            if torch.is_tensor(reward):
                reward = np.array(reward.cpu())
            if torch.is_tensor(next_state):
                next_state = np.array(next_state.cpu())
            if torch.is_tensor(done):
                done = np.array(done.cpu())
                        
            self.buffer.update_td_error(index,state,action,reward,next_state, done, td_error.item())
        
        
        loss = self.criterion(outputs.gather(1, actions), td_target)
        
        self.optimizer.zero_grad()
        
        loss.backward()
        
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)     
    
class PrioritizedReplayBuffer(agent.ReplayBuffer):
    """Fixed-size buffer to store experience tuples using a priority sampling method"""
    
    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.2, beta=0.9, constant=.1):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (int): parameter correcting how non-uniform the sampling is (closer to 0 -> completely non-uniform, closer to 1 -> completely uniform)
            beta (int): parameter describing the importance of the sampling weights in the update rule
            constant (int): parameter used to make non valuable items more likely to be picked
        """
        super().__init__(action_size, buffer_size, batch_size, seed)
        
        # used to determine sampling probability
        self.sampling_td_errors = deque(maxlen=buffer_size) 
        
        self.alpha = alpha
        self.beta = beta
        self.constant = constant
    
    def add(self, state, action, reward, next_state, done, td_error):
        
        if td_error < 0:
            td_error = - td_error
        
        self.sampling_td_errors.append(td_error)
        super().add(state, action, reward, next_state, done)
    
    def sample(self):
        
        """Randomly sample a batch of experiences from memory."""
        
        probabilities = [(p**self.alpha) for p in self.sampling_td_errors]
        
        sum_probs = sum(probabilities)
        
        probabilities = [(p/sum_probs) for p in probabilities]
        
        experience_indexes = np.random.choice(len(self.memory), size=self.batch_size, p=probabilities)
        
        experiences = []
        
        for index in experience_indexes:
            experiences.append(self.memory[index])
                    
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
            
        return (states, actions, rewards, next_states, dones, experience_indexes)

    def update_td_error(self, index, state, action, reward, next_state, done, td_error):
        
        if td_error < 0:
            td_error = - td_error
        
        e = self.experience(state, action, reward, next_state, done)
        self.memory[index] = e
        self.sampling_td_errors[index] = td_error
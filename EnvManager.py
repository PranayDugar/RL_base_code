import gymnasium as gym
import numpy as np
import torch
from Algo import Algorithm

class EnvManager:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
    def reset(self, tensor=False, tensor_device='cpu'):
        state = self.env.reset()
        if tensor:
            state = torch.tensor(state, dtype=torch.float32, device=tensor_device)
        return state
    
    def step(self, action):
        return self.env.step(action)
    
    def rollout(self, policy:Algorithm, num_episodes, max_steps=None, create_tensor=False, tensor_device=None):
        transitions = []
        for _ in range(num_episodes):
            state, _ = self.reset(tensor=True)
            done = False
            step_count = 0
            while not done:
                action = policy.get_action(state)
                next_state, reward, done, trun, _ = self.step(action)
                transitions.append((state, action, reward, next_state, done))
                state = next_state
                step_count += 1
                if max_steps is not None and step_count >= max_steps:
                    break
        
        if create_tensor:
            transitions = self.make_tensor(transitions, tensor_device)
                
        return transitions
    
        # <np arrays>
    def make_batch(self, transitions):
        create_batch = lambda index: np.array([t[index] for t in transitions])
        
        states_batch = create_batch(0)
        actions_batch = create_batch(1)
        rewards_batch = create_batch(2)
        next_states_batch = create_batch(3)
        dones_batch = create_batch(4)
        
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch

    def make_tensor(self, batch, device=None):
        if device is None:
            device = torch.device("cpu")
        return [torch.tensor(x, dtype=torch.float32, device=device) for x in batch]
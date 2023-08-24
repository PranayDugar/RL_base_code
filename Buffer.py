import torch
import numpy as np
class Buffer:
    def __init__(self, capacity, state_shape, action_shape):
        self.capacity = capacity
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.states = np.empty((capacity, state_shape), dtype=np.float32)
        self.actions = np.empty((capacity, action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.next_states = np.empty((capacity, state_shape), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=np.bool)
        self.index = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, states, actions, rewards, next_states, dones):
        assert len(states) == len(actions) == len(rewards) == len(next_states) == len(dones), "Batch sizes must be equal"
        
        num_to_add = len(states)
        remaining_capacity = self.capacity - self.size
        
        if num_to_add <= remaining_capacity:
            end_index = (self.index + num_to_add) % self.capacity
            self.states[self.index:end_index] = states
            self.actions[self.index:end_index] = actions
            self.rewards[self.index:end_index] = rewards
            self.next_states[self.index:end_index] = next_states
            self.dones[self.index:end_index] = dones
        else:
            overflow = num_to_add - remaining_capacity
            self.states[self.index:] = states[:remaining_capacity]
            self.actions[self.index:] = actions[:remaining_capacity]
            self.rewards[self.index:] = rewards[:remaining_capacity]
            self.next_states[self.index:] = next_states[:remaining_capacity]
            self.dones[self.index:] = dones[:remaining_capacity]
            
            self.states[:overflow] = states[remaining_capacity:]
            self.actions[:overflow] = actions[remaining_capacity:]
            self.rewards[:overflow] = rewards[remaining_capacity:]
            self.next_states[:overflow] = next_states[remaining_capacity:]
            self.dones[:overflow] = dones[remaining_capacity:]

        self.index = (self.index + num_to_add) % self.capacity
        self.size = min(self.size + num_to_add, self.capacity)

    def reset(self):
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_states.fill(0)
        self.dones.fill(False)
        self.index = 0
        self.size = 0
    
    def sample(self, batch_size):
        assert batch_size <= self.size, "Batch size must be smaller than buffer size"
        
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        sstates = self.states[indices]
        sactions = self.actions[indices]
        srewards = self.rewards[indices]
        snext_states = self.next_states[indices]
        sdones = self.dones[indices]
        
        return sstates, sactions, srewards, snext_states, sdones

    def sample_tensor(self, batch_size, device):
        assert batch_size <= self.size, "Batch size must be smaller than buffer size"
        batch = self.sample(batch_size)
        sstates, sactions, srewards, snext_states = [torch.tensor(x, dtype=torch.float32, device=device) for x in batch[:-1]]
        sdones = torch.tensor(batch[-1], dtype=torch.bool, device=device)
        return sstates, sactions, srewards, snext_states, sdones
    
    
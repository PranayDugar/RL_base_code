import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod

class Algorithm(ABC):
    def __init__(self, actor, critic, learning_rate_actor, learning_rate_critic, discount_factor):
        self.actor = actor
        self.critic = critic
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.discount_factor = discount_factor

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic)

    @abstractmethod
    def get_action_distribution(self, state):
        pass

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def get_value(self, state):
        pass

    @abstractmethod
    def compute_target(self, reward, next_state, done):
        pass

    @abstractmethod
    def compute_grads(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def update_grads(self, actor_loss, critic_loss):
        pass

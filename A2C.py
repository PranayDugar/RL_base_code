import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod
from Algo import Algorithm
from Buffer import Buffer
from EnvManager import EnvManager as Env
from Model import Actor, Critic

class A2C(Algorithm):
    def __init__(self, actor, critic, learning_rate_actor, learning_rate_critic, discount_factor):
        super().__init__(actor, critic, learning_rate_actor, learning_rate_critic, discount_factor)

    def get_action_distribution(self, state):
        """ Pass state through the actor and generate action distributions

        Args:
            state (np.array): set of states that have to be processed
        """
        with torch.no_grad():
            logits = self.actor(state)
            action_distribution = torch.distributions.Categorical(logits=logits)
        return action_distribution

    def get_max_action(self, state):
        """ Pass state through the actor and generate actions

        Args:
            state (np.array): set of states that have to be processed
        """
        with torch.no_grad():
            action_distribution = self.get_action_distribution(state)
            action = torch.argmax(action_distribution).item()
        return action
    
    def get_sample_action(self, state):
        """ Pass state through the actor and generate actions

        Args:
            state (np.array): set of states that have to be processed
        """
        with torch.no_grad():
            action_distribution = self.get_action_distribution(state)
            action = action_distribution.sample().item()
        return action

    def get_action(self, state, greedy=False):
        if not greedy:
            return self.get_sample_action(state)
        else:
            return self.get_max_action(state)
    
    def get_value(self, state):
        with torch.no_grad():
            value = self.critic(state)
        return value

    def compute_target(self, rewards, next_states, states, dones):
        """ 
        compute the delta for AC
        """
        delta = rewards + self.discount_factor * self.get_value(next_states) * (1 - dones) - self.get_value(states)
        return delta

    def compute_grads(self, state, action, reward, next_state, done):
        """ Compute gradients for the actor and critic networks

        Args:
            state (np.array): set of states that have to be processed
            action (int): action taken in the state
            reward (float): reward received from the environment
            next_state (np.array): next state that the environment transitioned to
            done (bool): whether the episode is done or not
        """
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        delta = self.compute_target(reward, next_state, state, done)

        actor_loss = -torch.log(self.actor(state)[action]) * delta
        critic_loss = delta ** 2

        return actor_loss, critic_loss

    def update_grads(self, actor_loss, critic_loss):
        """ Update gradients for the actor and critic networks

        Args:
            actor_loss (torch.tensor): loss for the actor network
            critic_loss (torch.tensor): loss for the critic network
        """
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def collect_data(self, env: Env, num_episodes, max_steps):
        """ Collect data from the environment

        Args:
            env (EnvManager): environment manager object
            num_episodes (int): number of episodes to run
            max_steps (int): maximum number of steps per episode
        """
        buffer = Buffer(1000, env.state_dim, env.action_dim)
        transition_data = env.rollout(self, num_episodes, max_steps, create_tensor=False)
        buffer.add_batch(*transition_data)
        return buffer
    
    def train(self, buffer: Buffer, batch_size=32):
        """ Train the actor and critic networks

        Args:
            buffer (Buffer): buffer object containing the data
        """
        for batch in buffer.sample(32):
            states, actions, rewards, next_states, dones = batch
            actor_loss, critic_loss = self.compute_grads(states, actions, rewards, next_states, dones)
            self.update_grads(actor_loss, critic_loss)
                          
    def main_loop(self, env, iterations, epochs, num_episodes, max_steps):
        """ Main loop for the algorithm that iterates multiple iteration of collection and training

        Args:
            env (EnvManager): environment manager object
            num_episodes (int): number of episodes to run
            max_steps (int): maximum number of steps per episode
        """
        for _ in range(iterations):
            buffer = self.collect_data(env, num_episodes, max_steps)
            for _ in range(epochs):
                self.train(buffer)
        

if __name__ == "__main__":
    env = Env("CartPole-v1")
    actor = Actor(env.state_dim, env.action_dim)
    critic = Critic(env.state_dim)
    algo = A2C(actor, critic, 0.001, 0.001, 0.99)
    algo.main_loop(env, 100, 10, 10, 100)
    print("Done")
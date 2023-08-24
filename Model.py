import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.actor(state)
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim

        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        return self.critic(state)

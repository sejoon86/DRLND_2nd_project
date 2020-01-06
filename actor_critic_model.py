import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)



class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden1=400,  hidden2=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.state_size = state_size
        self.action_size = action_size 
        
        self.fc1 = nn.Linear(self.state_size,                    hidden1)
        self.fc2 = nn.Linear(hidden1 + self.action_size,         hidden2)
        self.fc3 = nn.Linear(hidden2,                            1) 
        self.reset_parameters()
        
        self.bn1 = nn.BatchNorm1d(hidden1)
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        

    def forward(self, state, action): 
        """Build a network that maps state -> action values.""" 
        ##x = F.relu(self.fc1(state)) 
        x = F.relu(self.bn1(self.fc1(state))) 
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class Deterministic_Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden1=400,  hidden2=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Deterministic_Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.state_size = state_size
        self.action_size = action_size 
        
        self.fc1 = nn.Linear(self.state_size, hidden1)
        self.fc2 = nn.Linear(hidden1,         hidden2)
        self.fc3 = nn.Linear(hidden2,         self.action_size)
        self.reset_parameters()
        
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.bn3 = nn.BatchNorm1d(self.action_size)
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = F.relu(self.fc1(state)) 
        x = F.relu(self.fc2(x)) 
        x = F.tanh(self.fc3(x)) 
            
            
        ####x = F.relu(self.bn1(self.fc1(state)))
        ####x = F.relu(self.bn2(self.fc2(x)))
        ####x = torch.tanh(self.bn3(self.fc3(x)))
        ##x = torch.tanh(self.fc3(x))

        return x

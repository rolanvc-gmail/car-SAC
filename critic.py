import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class CriticNetwork(nn.Module):
    """
        The CriticNetwork takes a state-action and outputs a state-action value.
    """
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # input is expected to be (1,1,84,84)
        # after self.conv1, output should be (1,1,80,80). after self.conv2, output is (1,1, 19,19)
        self.conv1 = nn.Conv2d(1, 1, 5)
        self.conv2 = nn.Conv2d(1, 1, 3)  # we expect output of Conv to be (19x19)
        self.fc1 = nn.Linear((19*19) + n_actions, self.fc1_dims) # input is (19x19+3 = 361+4=365), output is 256
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)  # in is 256, out is 256
        self.q = nn.Linear(self.fc2_dims, 1)  # in is 256, out is 1

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        # state is Tensor(1,1,84,84)
        # self.conv1 filter size is 5. Therefore 84-5=79; 79+1=80
        x = self.conv1(state)  # x is Tensor (1,1,80,80), as expected.
        x = F.max_pool2d(x, 2)  # x is Tensor (1,1, 40,40), as expected.
        x = F.relu(x)  # x is Tensor( 1, 1, 40,40) as expected
        x = self.conv2(x)  # self.conv2 filter size is  3. therefore, output should be 40-3=37; 37+1=38
        x = F.max_pool2d(x, 2)  # x is Tensor(1,1,19,19)
        x = F.relu(x)  # x is Tensor(1,1,19,19)
        x = T.flatten(x, start_dim=2)  # x is Tensor(361,). 19*19=361

        # x is Tensor(1,1, 361), action is Tensor(1,1,3).
        # we try to squeeze them first...
        x = T.squeeze(x, dim=1)
        action = T.squeeze(action, dim=1)
        concatenated = T.cat([x, action], dim=1)  # concatentated is Tensor(16,364).
        action_value = self.fc1(concatenated)  # action_value is Tensor(16,256)
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        #  action_value is Tensor(16,3)
        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
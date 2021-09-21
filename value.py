import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class ValueNetwork(nn.Module):
    """
    The ValueNetwork takes a state and outputs (presumably) a value.

    """
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
            name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        input_dims = (18, 18)
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # input is (1,1,84,84)
        self.conv1 = nn.Conv2d(1, 1, 5)             # after self.conv1, output is (1,1,80,80)
        self.conv2 = nn.Conv2d(1, 1, 3)             # after self.conv2, output is (1,1,19,19)
        self.fc1 = nn.Linear(361, self.fc1_dims)  # input will be flattened 19x19 = 361,
                                                  # output will be 256
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims) # input is 256; output is 256
        self.v = nn.Linear(self.fc2_dims, 1)  # input is 256;output is 1

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):  # state is Tensor(256,1,84,84)
        x = self.conv1(state)  # x is Tensor (1,1,80,80), as expected.
        x = F.max_pool2d(x, 2)  # x is Tensor (1,1, 40,40), as expected.
        x = F.relu(x)  # x is Tensor( 1, 1, 40,40) as expected
        x = self.conv2(x)  # self.conv2 filter size is  3. therefore, output should be 40-3=37; 37+1=38
        x = F.max_pool2d(x, 2)  # x is Tensor(256,1,19,19)
        x = F.relu(x)  # x is Tensor(256,1,19,19)

        # x = x.view((256, 361))  # x is Tensor(256,1,19,19). I want to convert it to (256,361). Maybe I shouldn't use views()?
        x = x.flatten(start_dim=1)
        state_value = self.fc1(x)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)  # state_value is Tensor(256,1,84,256).  the output of the v layer should be Tensor(256,1).

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


class ActorNetwork(nn.Module):
    """
    The ActionNetwork takes a state and outputs an action.
    """
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256,
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        input_dims = (18, 18)
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        #input is (batch, 1, 84,84)
        self.conv1 = nn.Conv2d(1, 1, 5)
        self.conv2 = nn.Conv2d(1, 1, 3)
        # after self.conv1, output should be (batch,1, 40,40). after self.conv2, output should be (b, 1, 19,19)
        # will flatten here to be (batch, 1, 361)
        self.fc1 = nn.Linear(361, self.fc1_dims) # input should be (batch, 361), output is (batch, 256)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        # state is Tensor(1,1,84,84)
        # self.conv1 filter size is 5. Therefore 84-5=79; 79+1=80
        x = self.conv1(state)  # x is Tensor (1,1,80,80), as expected.
        x = F.max_pool2d(x, 2)  # x is Tensor (1,1, 40,40), as expected.
        x = F.relu(x)  # x is Tensor( 1, 1, 40,40) as expected
        x = self.conv2(x)  # self.conv2 filter size is  3. therefore, output should be 40-3=37; 37+1=38
        x = F.max_pool2d(x, 2)  # x is Tensor(1,1,19,19)
        x = F.relu(x)  # x is Tensor(1,1,19,19)

        # When x is Tensor(1,1,19,19) we want a result of Tensor(1,1, 361).
        # When x is Tensor(256,1,19,19) we want a result of Tensor(256,1,361).
        x = x.flatten(start_dim=1)

        prob = self.fc1(x)  # fc1 input is (19,19) output is 256
        prob = F.relu(prob)  # prob is Tensor(256,)
        prob = self.fc2(prob)  # fc2 input is 256
        prob = F.relu(prob)  # prob is Tensor(256,)

        # mu and sigma are Linear(fc2_dims, n_actions)
        mu = self.mu(prob)  # prob is Tensor(3,)
        sigma = self.sigma(prob) # sigma is Tensor(3,)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):  # state is Tensor(256,1, 84,84)
        mu, sigma = self.forward(state)  # mu and sigma are both Tensor(1,1,18,3)
        probabilities = Normal(mu, sigma)  # pobabilities is Normal(loc: size(1,1,18,3), scale: size(1,1,18,13)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        #  at this point, actions is Tensor(256,3).
        temp = T.tensor(self.max_action).to(self.device)
        action = T.tanh(actions) * temp # action is Tensor(256,3)---good?.
        log_probs = probabilities.log_prob(actions)  # log_probs is Tensor(256,3)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)  # still Tensor(256,3)
        log_probs = log_probs.sum(1, keepdim=True)
        # log_probs = log_probs.sum()

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

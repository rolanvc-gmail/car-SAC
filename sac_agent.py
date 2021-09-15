import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from critic import CriticNetwork
from value import ValueNetwork
from actor import ActorNetwork
from torchviz import make_dot


class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=(1, 84, 84),
            env=None, gamma=0.99, n_actions=2, max_size=100000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                    name='actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_2')
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)  # actions is Tensor(1, 1,18,3)

        return actions.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        """"
          Step 1: Sample from the replay buffer.
        """

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        """
            Step 2: Compute the values of the current and next state
        """
        value = self.value(state)
        value_ = self.target_value(state_).view(-1)  # value_ is Tensor(21504,)
        value_[done] = 0.0 #  the size of value_ is Tensor(21504,). done is of size Tensor(256,)d

        """
            Step 3: Predict the actions  and log probabilities of the current state.
        """
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)  # state is Tensor(256,1,84,84)
                                                            # action is Tensor(256,3) log_probs is Tensor(256,1)
         # log_probs = log_probs.view(-1)  # log_probs is Tensor(256,)
        q1_new_policy = self.critic_1.forward(state, actions)  # q1_new_policy is Tensor(256,1)
        q2_new_policy = self.critic_2.forward(state, actions)  # q2_new_policy is Tensor(256,1)
        critic_value = T.min(q1_new_policy, q2_new_policy)  # critic_value is Tensor(256,1)
        # critic_value = critic_value.view(-1)  # critic_value is now Tensor(256,)

        """
            Step 4: Backpropagate the value network. 
        """
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs # value_target is Tensor(256,1)
        the_mse_loss = F.mse_loss(value, value_target, reduction='none')  # the_mse_loss is Tensor(256,1)
        value_loss = 0.5 * the_mse_loss
        value_loss = value_loss.sum()
        dot = make_dot(value_loss)
        dot.render("network.png")
        value_loss.backward(retain_graph=True)  #value_loss is Tensor(256,1), keep getting error here. why?
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)

        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()


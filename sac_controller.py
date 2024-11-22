import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, action_scale):
        """
        SAC Actor network.
        Predicts mean and log standard deviation of actions.
        """
        super(ActorNetwork, self).__init__()
        self.action_scale = action_scale

        # Actor network layers
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 400)
        self.mean = nn.Linear(400, action_size)
        self.log_std = nn.Linear(400, action_size)

    def forward(self, state):
        """
        Forward pass through the network.
        Returns the mean and log standard deviation for the action distribution.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        """
        Sample an action from the policy.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        SAC Critic network.
        Predicts Q-values for state-action pairs.
        """
        super(CriticNetwork, self).__init__()

        # Q1 network
        self.fc1_1 = nn.Linear(state_size + action_size, 400)
        self.fc1_2 = nn.Linear(400, 400)
        self.fc1_3 = nn.Linear(400, 1)

        # Q2 network
        self.fc2_1 = nn.Linear(state_size + action_size, 400)
        self.fc2_2 = nn.Linear(400, 400)
        self.fc2_3 = nn.Linear(400, 1)

    def forward(self, state, action):
        """
        Compute Q1 and Q2 values.
        """
        xu = torch.cat([state, action], 1)

        # Q1 network forward
        x1 = F.relu(self.fc1_1(xu))
        x1 = F.relu(self.fc1_2(x1))
        x1 = self.fc1_3(x1)

        # Q2 network forward
        x2 = F.relu(self.fc2_1(xu))
        x2 = F.relu(self.fc2_2(x2))
        x2 = self.fc2_3(x2)

        return x1, x2

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            np.asarray(done, dtype=np.float32)
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(
            np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class SACController:
    def __init__(self, state_size, action_size, a_max, training=True, device='cpu'):
        """
        Initialize the SAC controller.
        """
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.a_max = a_max
        self.training = training

        # Initialize networks
        self.policy_net = ActorNetwork(state_size, action_size, a_max).to(device)
        self.critic_net = CriticNetwork(state_size, action_size).to(device)
        self.critic_net_target = CriticNetwork(state_size, action_size).to(device)
        self.critic_net_target.load_state_dict(self.critic_net.state_dict())

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=3e-4)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=1_000_000)

        # SAC parameters
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2  # Entropy coefficient
        self.target_entropy = -action_size  # Target entropy

        # Automatic entropy tuning
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

    def compute_force(self, state):
        """
        Compute control force using the SAC policy.
        """
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action_tensor, _ = self.policy_net.sample(state_tensor)
        action = action_tensor.detach().cpu().numpy()[0]  # Now action is an array of shape (action_size,)
        return action 

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in the replay buffer.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update_parameters(self, batch_size):
        """
        Update the SAC networks using the replay buffer.
        """
        if len(self.replay_buffer) < batch_size:
            return

        # Sample a batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        # Compute target Q value
        with torch.no_grad():
            next_action, next_log_prob = self.policy_net.sample(next_state_batch)
            target_q1, target_q2 = self.critic_net_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q_value = reward_batch + (1 - done_batch) * self.gamma * target_q

        # Update critic network
        current_q1, current_q2 = self.critic_net(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q1, target_q_value) + F.mse_loss(current_q2, target_q_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor network
        new_action, log_prob = self.policy_net.sample(state_batch)
        q1_new, q2_new = self.critic_net(state_batch, new_action)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        # Update alpha (entropy coefficient)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        # Soft update of target networks
        for target_param, param in zip(self.critic_net_target.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

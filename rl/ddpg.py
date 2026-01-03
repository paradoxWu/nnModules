import random
from collections import deque
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-4):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.tanh = nn.Tanh()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate=1e-4):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(torch.cat([x, action], dim=1)))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim, buffer_size, batch_size, actor_lr, critic_lr, tau, gamma):
        self.actor = Actor(state_dim, hidden_dim, action_dim, actor_lr)
        self.actor_target = Actor(state_dim, hidden_dim, action_dim, actor_lr)
        self.critic = Critic(state_dim, action_dim, hidden_dim, critic_lr)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim, critic_lr)

        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma

        self._update_target_networks(tau=1)  # initialize target networks

    def act(self, state, noise=0.0):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        return np.clip(action + noise, -1, 1)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Update Critic
        self.critic.optimizer.zero_grad()

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q_values = self.critic_target(next_states, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * target_q_values

        current_q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        critic_loss.backward()
        self.critic.optimizer.step()

        # Update Actor
        self.actor.optimizer.zero_grad()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        self._update_target_networks()

    def _update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def train_ddpg(env, agent, episodes, max_steps, num_agents, noise_scale=0.1, noise_decay=0.99):
    scores_window = deque(maxlen=100)
    scores = []

    for episode in range(1, episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent_scores = np.zeros(num_agents)

        for step in range(max_steps):
            actions = agent.act(states, noise_scale)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            for i in range(num_agents):
                agent.store_transition(states[i], actions[i], rewards[i], next_states[i], dones[i])
            agent.learn()

            states = next_states
            agent_scores += rewards
            noise_scale *= noise_decay

            if np.any(dones):
                break

        avg_score = np.mean(agent_scores)
        scores_window.append(avg_score)
        scores.append(avg_score)

        if episode % 10 == 0:
            print(f"Episode: {episode}, Score: {avg_score:.2f}, Avg Score: {np.mean(scores_window):.2f}")

        # Saving trained Networks
        torch.save(agent.actor.state_dict(), "actor_final.pth")
        torch.save(agent.critic.state_dict(), "critic_final.pth")

    return scores

if __name__ == "__main__":
    env = UnityEnvironment(file_name='Reacher_20.app')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    state_dim = 33
    action_dim = brain.vector_action_space_size

    num_agents = 20
    
    # Hyperparameter suggestions
    hidden_dim = 400
    batch_size = 256
    actor_lr = 5e-4
    critic_lr = 1e-3
    tau = 5e-3
    gamma = 0.995
    noise_scale = 0.5
    noise_decay = 0.998

    agent = DDPG(state_dim, action_dim, hidden_dim=hidden_dim, buffer_size=200000, batch_size=batch_size,
                 actor_lr=actor_lr, critic_lr=critic_lr, tau=tau, gamma=gamma)

    episodes = 200
    max_steps = 1000

    scores = train_ddpg(env, agent, episodes, max_steps, num_agents, noise_scale=0.2, noise_decay=0.995)

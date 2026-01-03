import torch
import torch.nn as nn
import numpy as np

# import gym
import gymnasium as gym
import gymnasium_robotics as gym_robot

# https://developer.aliyun.com/article/1208127

# https://blog.csdn.net/gitblog_00340/article/details/151603984
# 1. 策略模型：直接输出机器人关节控制指令
class RobotPolicyModel(nn.Module):
    def __init__(self, state_dim, action_dim, action_high, action_low):
        super().__init__()
        self.action_high = torch.tensor(action_high)
        self.action_low = torch.tensor(action_low)

        self.network = nn.Sequential(
            nn.Linear(state_dim, 512),  # 增加隐藏层容量
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        # 将输出缩放到实际动作空间（如关节扭矩[-2, 2] Nm）
        action = self.network(state)
        return self.action_low + (action + 1) * 0.5 * (
            self.action_high - self.action_low
        )

    def get_action(self, state):
        # 在机器人训练中通常添加探索噪声
        mean = self.forward(state)
        # 使用对角高斯分布进行探索
        dist = torch.distributions.Normal(mean, torch.ones_like(mean) * 0.1)
        return dist.sample()


# 2. 价值模型：评估机器人当前状态的好坏
class RobotValueModel(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),  # 输出状态价值
        )

    def forward(self, state):
        return self.network(state)


# 3. 奖励模型：在机器人训练中通常由任务目标定义
class RobotRewardModel:
    def __init__(self):
        # 机器人任务特定的奖励权重
        self.weights = {
            "task_completion": 10.0,  # 任务完成奖励
            "energy_penalty": 0.1,  # 能量消耗惩罚
            "collision_penalty": 100.0,  # 碰撞惩罚
        }

    def compute_reward(self, obs, action, next_obs, done):
        """
        机器人状态通常包含：
        - obs[:n]: 关节位置
        - obs[n:2n]: 关节速度
        - obs[2n:]: 末端执行器位置、目标位置等
        """
        reward = 0

        # 任务奖励：例如机械臂到达目标位置
        ee_pos = next_obs[-6:-3]  # 末端执行器位置
        target_pos = next_obs[-3:]  # 目标位置
        distance = np.linalg.norm(ee_pos - target_pos)
        reward += self.weights["task_completion"] * np.exp(-distance)

        # 能量惩罚：抑制剧烈动作
        energy = np.sum(action**2)
        reward -= self.weights["energy_penalty"] * energy

        # 碰撞惩罚：避免自碰撞或环境碰撞
        if self.check_collision(next_obs):
            reward -= self.weights["collision_penalty"]
            done = True

        return reward, done

    def check_collision(self, obs):
        # 简化的碰撞检测逻辑
        # 实际中会调用仿真环境的碰撞检测API
        return False


# 4. Reference Model：在机器人训练中用于行为克隆或约束策略不偏离太多
class ReferenceModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 通常是预训练的策略或专家策略
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(), nn.Linear(256, action_dim)
        )

    def forward(self, state):
        return self.network(state)


class RobotPPOTrainer:
    def __init__(self, env_name="Reacher-v2"):
        # 创建MuJoCo/Gym环境
        self.env = gym.make(env_name)

        # 获取机器人状态/动作维度
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_high = self.env.action_space.high
        self.action_low = self.env.action_space.low

        # 初始化模型
        self.policy = RobotPolicyModel(
            self.state_dim, self.action_dim, self.action_high, self.action_low
        )
        self.value = RobotValueModel(self.state_dim)
        self.reward_model = RobotRewardModel()
        self.reference = ReferenceModel(self.state_dim, self.action_dim)

        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=1e-3)

        # PPO超参数
        self.gamma = 0.99
        self.lamda = 0.95  # GAE参数
        self.clip_ratio = 0.1
        self.target_kl = 0.01

    def collect_trajectories(self, num_steps=64):
        """在仿真环境中收集机器人交互数据"""
        states, actions, rewards, values, log_probs = [], [], [], [], []

        state = self.env.reset()[0]
        for step in range(num_steps):
            with torch.no_grad():
                # 1. 策略模型生成动作（机器人关节指令）
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = self.policy.get_action(state_tensor)
                action_np = action.numpy()[0]

                # 2. 在仿真环境中执行动作
                next_state, env_reward, done, _, dict = self.env.step(action_np)

                # 3. 计算奖励（可结合任务奖励模型）
                task_reward, done = self.reward_model.compute_reward(
                    state, action_np, next_state, done
                )

                # 4. 价值模型评估状态
                value = self.value(state_tensor)

                # 5. 记录数据
                states.append(state)
                actions.append(action_np)
                rewards.append(task_reward)
                values.append(value.item())
                log_prob = (
                    torch.distributions.Normal(
                        self.policy(state_tensor), torch.ones_like(action) * 0.1
                    )
                    .log_prob(action)
                    .sum()
                )
                log_probs.append(log_prob.item())

                state = next_state
                if done:
                    state = self.env.reset()[0]

        return {
            "states": torch.FloatTensor(states),
            "actions": torch.FloatTensor(actions),
            "rewards": torch.FloatTensor(rewards),
            "values": torch.FloatTensor(values),
            "log_probs": torch.FloatTensor(log_probs),
        }

    def compute_gae(self, rewards, values, dones=None):
        """计算广义优势估计（GAE）"""
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # 最后一个状态的下一状态价值为0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lamda * last_gae

        returns = advantages + values
        return advantages, returns

    def update_policy(self, batch):
        """PPO策略更新"""
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]

        # 计算优势函数
        advantages, returns = self.compute_gae(
            batch["rewards"], batch["values"],)
            #   batch["dones"])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(10):  # PPO的epoch数
            # 1. 当前策略的新log_prob
            mean = self.policy(states)
            dist = torch.distributions.Normal(mean, torch.ones_like(mean) * 0.1)
            new_log_probs = dist.log_prob(actions).sum(dim=1)

            # 2. PPO的clip目标函数
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # 3. 添加KL散度约束（防止策略偏离过大）
            with torch.no_grad():
                ref_actions = self.reference(states)
                kl_div = (
                    torch.distributions.kl_divergence(
                        torch.distributions.Normal(mean, 0.1),
                        torch.distributions.Normal(ref_actions, 0.1),
                    )
                    .sum(dim=1)
                    .mean()
                )

            # 如果KL散度过大，提前终止
            if kl_div > 1.5 * self.target_kl:
                break

            # 4. 价值函数损失
            values = self.value(states).squeeze()
            value_loss = torch.nn.functional.mse_loss(values, returns)

            # 5. 反向传播更新
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

    def train(self, total_timesteps=1000000):
        """主训练循环"""
        timestep = 0

        while timestep < total_timesteps:
            # 1. 收集机器人仿真数据
            batch = self.collect_trajectories()

            # 2. PPO更新
            self.update_policy(batch)

            timestep += len(batch["states"])

            # 3. 评估策略
            if timestep % 1280 == 0:
                avg_reward = self.evaluate()
                print(f"Timestep: {timestep}, Avg Reward: {avg_reward:.2f}")

    def evaluate(self, num_episodes=10):
        """评估机器人策略性能"""
        total_rewards = 0

        for _ in range(num_episodes):
            state = self.env.reset()[0]
            done = False
            episode = 0
            while not done:
                episode += 1
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = self.policy(state_tensor)
                action_np = action.detach().numpy()[0]
                # 执行动作
                state, reward, done, _,dict = self.env.step(action_np)
                total_rewards += reward
                if(episode > num_episodes):
                    break

        return total_rewards / num_episodes


if __name__ == "__main__":
    # envs = gym.envs.registry.keys()
    # for env in envs:
    #     print(env)
    # 运行训练
    trainer = RobotPPOTrainer("Reacher-v5")
    trainer.train()

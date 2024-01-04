import numpy as np
from collections import deque


class ClassicalAgent:
    def __init__(self, num_states, num_actions, step_size, epsilon, discount):
        # NOTE: Assume discrete state and action representation starting from 0.
        # NOTE: Assume actions are the same for all states.
        self.num_states = num_states
        self.num_actions = num_actions
        self.step_size = step_size
        self.epsilon = epsilon
        self.discount = discount
        self.q_table = np.random.rand(num_states, num_actions)

    def uniform_sample(self):
        return np.random.choice(self.num_actions)

    def epsilon_greedy(self, obs):
        p = np.random.rand()
        if p < self.epsilon:
            return self.uniform_sample()
        else:
            return np.argmax(self.q_table[obs])

    def sample_action(self, obs):
        return self.epsilon_greedy(obs)

    def update(self):
        pass

    def reset(self):
        # self.q_table = np.random.rand(self.num_states, self.num_actions)
        pass

    def learn(self, env, step_limit):
        obs, info = env.reset()
        reward_per_episode = []
        sum_of_reward = 0

        for _ in range(step_limit):
            action = self.sample_action(
                obs
            )  # agent policy that uses the observation and info
            next_obs, reward, terminated, truncated, info = env.step(action)
            self.update(obs, action, reward, next_obs, terminated or truncated)
            sum_of_reward += reward
            obs = next_obs

            if terminated or truncated:
                obs, info = env.reset()
                self.reset()
                reward_per_episode.append(sum_of_reward)
                sum_of_reward = 0

        return reward_per_episode


class SARSA(ClassicalAgent):
    def __init__(self, num_states, num_actions, step_size, epsilon, discount):
        super().__init__(num_states, num_actions, step_size, epsilon, discount)
        self.next_action = None

    def sample_action(self, obs):
        if self.next_action == None:
            return self.epsilon_greedy(obs)
        else:
            return self.next_action

    def reset(self):
        # super().reset()
        self.next_action = None

    def update(self, obs, action, reward, next_obs, done=None):
        # NOTE: Epsilon-greedy policy is used as behavior policy here.
        self.next_action = self.epsilon_greedy(next_obs)
        self.q_table[obs, action] += self.step_size * (
            reward
            + self.discount * self.q_table[next_obs, self.next_action]
            - self.q_table[obs, action]
        )


class Q_Learning(ClassicalAgent):
    def __init__(self, num_states, num_actions, step_size, epsilon, discount):
        super().__init__(num_states, num_actions, step_size, epsilon, discount)

    def update(self, obs, action, reward, next_obs, done=None):
        self.q_table[obs, action] += self.step_size * (
            reward
            + self.discount * np.max(self.q_table[next_obs])
            - self.q_table[obs, action]
        )

class Expected_SARSA(ClassicalAgent):
    def __init__(self, num_states, num_actions, step_size, epsilon, discount):
        super().__init__(num_states, num_actions, step_size, epsilon, discount)
        self.high_prob = 1-self.epsilon+self.epsilon/num_actions
        self.low_prob = self.epsilon/num_actions

    def update(self, obs, action, reward, next_obs, done=None):
        probs = np.repeat([self.low_prob], self.num_actions)
        probs[np.argmax(self.q_table[next_obs])] = self.high_prob
        self.q_table[obs, action] += self.step_size * (
            reward
            + self.discount * np.sum(probs*self.q_table[next_obs])
            - self.q_table[obs, action]
        )

class N_SARSA(ClassicalAgent):
    def __init__(
        self, num_states, num_actions, step_size, epsilon, discount, n_step
    ):
        super().__init__(num_states, num_actions, step_size, epsilon, discount)
        self.n_step = n_step
        self.reward_buffer = deque([], maxlen=self.n_step)
        self.obs_buffer = deque([], maxlen=self.n_step + 1)
        self.action_buffer = deque([], maxlen=self.n_step + 1)

    def reset(self):
        self.reward_buffer = deque([], maxlen=self.n_step)
        self.obs_buffer = deque([], maxlen=self.n_step + 1)
        self.action_buffer = deque([], maxlen=self.n_step + 1)

    def sample_action(self, obs):
        if len(self.action_buffer):
            return self.action_buffer[-1]
        else:
            return self.epsilon_greedy(obs)

    def update(self, obs, action, reward, next_obs, done):
        if not done:
            if not len(self.action_buffer):
                self.action_buffer.append(action)
            if not len(self.obs_buffer):
                self.obs_buffer.append(obs)

            self.reward_buffer.append(reward)
            self.obs_buffer.append(next_obs)
            next_action = self.epsilon_greedy(next_obs)
            self.action_buffer.append(next_action)

            if len(self.reward_buffer) == self.n_step:
                g = (self.discount**self.n_step) * self.q_table[
                    next_obs, next_action
                ]
                for i, r in enumerate(self.reward_buffer):
                    g += (self.discount**i) * r
                self.q_table[
                    self.obs_buffer[0], self.action_buffer[0]
                ] += self.step_size * (
                    g - self.q_table[self.obs_buffer[0], self.action_buffer[0]]
                )
        else:
            for i in range(self.n_step):
                g = 0
                for j in range(i, self.n_step):
                    g += (self.discount ** (j - i)) * self.reward_buffer[j]
                self.q_table[
                    self.obs_buffer[i], self.action_buffer[i]
                ] += self.step_size * (
                    g - self.q_table[self.obs_buffer[i], self.action_buffer[i]]
                )


class Tree_Backup(N_SARSA):
    def __init__(
        self, num_states, num_actions, step_size, epsilon, discount, n_step
    ):
        super().__init__(
            num_states, num_actions, step_size, epsilon, discount, n_step
        )
    
    def update(self, obs, action, reward, next_obs, done):
        if not done:
            if not len(self.action_buffer):
                self.action_buffer.append(action)
            if not len(self.obs_buffer):
                self.obs_buffer.append(obs)

            self.reward_buffer.append(reward)
            self.obs_buffer.append(next_obs)
            next_action = self.epsilon_greedy(next_obs)
            self.action_buffer.append(next_action)

            if len(self.reward_buffer) == self.n_step:
                # NOTE: Assume greedy policy (i.e., Q Learning)
                g = self.reward_buffer[-1] + self.discount*np.max(self.q_table[self.obs_buffer[-1]])
                for i in range(self.n_step-2, -1, -1):
                    if np.argmax(self.q_table[self.obs_buffer[i+1]])==self.action_buffer[i+1]:
                        g = self.reward_buffer[i] + self.discount*g
                    else:
                        g = self.reward_buffer[i] + self.discount*np.max(self.q_table[self.obs_buffer[i+1]])
                self.q_table[
                    self.obs_buffer[0], self.action_buffer[0]
                ] += self.step_size * (
                    g - self.q_table[self.obs_buffer[0], self.action_buffer[0]]
                )
        else:
            for i in range(self.n_step):
                g = self.reward_buffer[-1]
                for j in range(self.n_step-2, i-1, -1):
                    if np.argmax(self.q_table[self.obs_buffer[j+1]])==self.action_buffer[j+1]:
                        g = self.reward_buffer[j] + self.discount*g
                    else:
                        g = self.reward_buffer[j] + self.discount*np.max(self.q_table[self.obs_buffer[j+1]])
                self.q_table[
                    self.obs_buffer[i], self.action_buffer[i]
                ] += self.step_size * (
                    g - self.q_table[self.obs_buffer[i], self.action_buffer[i]]
                )

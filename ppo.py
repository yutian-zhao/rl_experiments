from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class PPO(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        device: torch.device,
        critic_lr: float,
        actor_lr: float,
        gamma: float,
        ent_coef: float,
        epsilon: float,
        n_envs: int,
    ) -> None:
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.epsilon = epsilon

        critic_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # estimate V(s)
        ]

        critic_target_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # estimate V(s)
        ]

        actor_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(
                32, n_actions
            ),  # estimate action logits (will be fed into a softmax later)
        ]

        # actor_k_layers = [
        #     nn.Linear(n_features, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(
        #         32, n_actions
        #     ),  # estimate action logits (will be fed into a softmax later)
        # ]

        # define actor and critic networks
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)
        # self.critic_target = nn.Sequential(*critic_target_layers).to(self.device)
        # NOTE: actor k to store actor's parameters from the last iteration
        # self.actor_k = nn.Sequential(*actor_k_layers).to(self.device)
        # self.critic_target.load_state_dict(self.critic.state_dict())

        # define optimizers for actor and critic
        # NOTE: from RMS to Adam, epsilon
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=actor_lr)

        self.debug_once = False

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.Tensor(x).to(self.device)
        state_values = self.critic(x)  # shape: [n_envs, 1]
        # NOTE: use actor_k to select action
        action_logits = self.actor(x)  # TODO:shape: [n_envs, n_actions] actor_k?
        return (state_values, action_logits)

    def select_action(
        self, x: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_values, action_logits = self.forward(x)
        action_pd = torch.distributions.Categorical(
            logits=action_logits
        )  # implicitly uses softmax
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()
        return (actions, action_log_probs, state_values, entropy)

    def get_losses(
        self,
        actions,
        states,
        action_k_log_probs,
        advantages,
        returns,
        masks,
        rewards,
        next_states,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # calculate the loss of the minibatch for actor and critic

        # critic_loss = advantages.pow(2).mean()
        # NOTE: in the original implementation, no detach
        # critic_loss = (self.critic(states)-rewards-gamma*masks*self.critic(next_states).detach()).pow(2).mean()
        pred_value = self.critic(states)
        critic_loss = F.mse_loss(pred_value, returns) # huber

        action_logits = self.actor(states)
        action_pd = torch.distributions.Categorical(
            logits=action_logits
        )  # implicitly uses softmax
        action_log_probs = action_pd.log_prob(torch.squeeze(actions)).unsqueeze(dim=-1)
        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(action_log_probs - action_k_log_probs)
        if self.debug_once:
            print(action_k_log_probs)
            print(action_log_probs)
            print(action_log_probs - action_k_log_probs)
            print(ratio)
            self.debug_once = False
        # clipped surrogate loss
        # when the reward is high, the prob of the action is small, the loss is large. loss=(-logp)*(return-baseline)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = -torch.minimum(policy_loss_1, policy_loss_2).mean()
        entropy = action_pd.entropy()
        actor_loss = (
            policy_loss - self.ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)
    

    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor, TAU=0.005
    ) -> None:
        actor_loss.backward()
        # torch.nn.utils.clip_grad_value_(self.actor.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()
        self.actor_optim.zero_grad()
        
        critic_loss.backward()
        # torch.nn.utils.clip_grad_value_(self.critic.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()
        self.critic_optim.zero_grad()

        # critic_target_state_dict = self.critic_target.state_dict()
        # critic_state_dict = self.critic.state_dict()
        # for key in critic_state_dict:
        #     critic_target_state_dict[key] = critic_state_dict[key] * \
        #         TAU + critic_target_state_dict[key]*(1-TAU)
        # self.critic_target.load_state_dict(critic_target_state_dict)


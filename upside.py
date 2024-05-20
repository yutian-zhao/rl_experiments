from utils import *
from models import *
from stable_baselines3 import DQN
import gym_examples
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class Node:
    # Question: why children=[] creates shared list
    def __init__(
        self,
        key,
        value,
        parent,
    ):
        self.parent = parent
        self.children = []
        self.value = value
        self.key = key

    def add(self, child):
        self.children.append(child)

    # def reindex(self):
    #     new_key = self.key
    #     queue = deque([self])
    #     while len(queue):
    #         node = queue.popleft()
    #         node.key = new_key
    #         new_key += 1
    #         for child in node.children:
    #             queue.append(child)
    #     return new_key # num_skill, max_skill

    def get_mapping(self):
        index = 0
        mapping = {}
        queue = deque([self])
        while len(queue):
            node = queue.popleft()
            if node.parent:
                mapping[node.key] = index
                index += 1
            for child in node.children:
                queue.append(child)
        return mapping

    def count(self):
        # return number of nodes in a tree, including root
        count = 1
        for child in self.children:
            count += child.count()
        return count

    def find(self, key):
        if key == self.key:
            return self
        else:
            for child in self.children:
                ret = child.find(key)
                if ret:
                    return ret
        return None

    def remove(self, key):
        child = self.find(key)
        if child:
            child.parent.children.remove(child)
            for c in child.children:
                if c.parent == child:
                    del c
            del child

    def print(self):
        string = str(self.key) + ",|\n"
        queue = deque([self])
        new_queue = deque([])
        while len(queue):
            node = queue.popleft()
            string += str(node.key) + ":"
            for child in node.children:
                new_queue.append(child)
                string += str(child.key) + ","
            if not len(node.children):
                string += "x"
            string += "|"
            if not len(queue):
                string += "\n"
                queue = new_queue
                new_queue = deque([])
        return string

    def __str__(self):
        return str(self.key)

    def __repr__(self):
        return str(self.key)


class DiscriminatorDataset(Dataset):
    def __init__(self, buffer, mapping):
        self.data = []
        for key, value in buffer.items():
            for v in value:
                self.data.append((v, mapping[key]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def rollout(env, policy, steps, obs=None):
    # return obs after n steps
    if obs is None:
        obs = env.reset()
    for _ in range(steps):
        action, _states = policy.predict(obs, deterministic=False)
        obs, reward, _, info = env.step(action)
        # if terminated or truncated:
        #     obs = policy.env.reset()
    return obs


def serial_rollout(env, policies, H, T):
    # return obs after (n+1)*steps for n policies
    obs = env.reset()
    for policy in policies[:-1]:
        obs = rollout(policy, T, obs=obs)
    return rollout(policies[-1], H, obs=obs)


def policy_only_serial_rollout(env, policies, T):
    # return obs after (n+1)*steps for n policies
    obs = env.reset()
    for policy in policies:
        obs = rollout(env, policy, T, obs=obs)
    return obs


def sample_diffuse(node, H, T, num):
    # sample num states in the diffuse part of a node
    states = []
    for _ in range(num):
        sample_policy_init_state(node, T)
        for _ in range(H):
            action = node.value.env.action_space.sample()
            state, _, terminated_truncated, info = node.value.env.step([action]) # vec_env
            if terminated_truncated:
                # TODO: handle vec_env
                state = info[0]['terminal_observation']
        states.append(state)
    return states


def sample_policy_init_state(node, T):
    # sample num states in the diffuse part of a node
    skills = []
    node_pointer = node
    while node_pointer.parent:
        skills.append(node_pointer.value)
        node_pointer = node_pointer.parent
    # skills = skills[:-1] # remove root
    skills = skills[::-1]
    state = policy_only_serial_rollout(node.value.env, skills, T)
    return state


def train_discriminator(
    discriminator, optimizer, dataset, epochs, bs, weight=None
):
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [0.8, 0.2]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=bs, shuffle=True)
    stopper = EarlyStopper(patience=500, min_delta=-0.001, if_save=True)
    training_losses = []
    valid_losses = []

    for epoch in tqdm(range(epochs)):
        num_batch = 0
        train_loss_sum = 0
        discriminator.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            X = X.float()
            # Compute prediction error
            pred = discriminator(X)
            # print(pred.shape, y.shape)
            loss = torch.nn.functional.cross_entropy(pred, y, weight=weight)
            num_batch += 1
            train_loss_sum += loss

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        training_losses.append(train_loss_sum / num_batch)

        discriminator.eval()
        valid_loss_sum = 0
        num_batch = 0
        for batch, (X, y) in enumerate(valid_dataloader):
            # TODO: check datatype
            X, y = X.to(device), y.to(device)
            X = X.float()

            # Compute prediction error
            pred = discriminator(X)
            loss = torch.nn.functional.cross_entropy(pred, y) # , weight=weight
            num_batch += 1
            valid_loss_sum += loss

        valid_loss = valid_loss_sum / num_batch
        valid_losses.append(valid_loss)

        if stopper.early_stop(
            valid_loss,
            state_dict=discriminator.state_dict(),
            path="upside_discrim_temp.dict",
        ):
            discriminator.load_state_dict(
                torch.load("upside_discrim_temp.dict")
            )
            break
    return valid_loss


def compute_discrim(discriminator, input, target):
    discriminator.eval()
    return torch.mean(
        F.softmax(discriminator(input), dim=1)[:, target]
    )  # [B, C] -> [B] -> [1]


def policy_learning(state_buffer, root, nodes, discrim_threhold, H, T, device):
    # parameters
    step_limit = 100000
    p2d_ratio = 5
    discrim_epoch = 1000
    iterations = 50
    discrim_lr = 1e-2
    discrim_bs = 64

    num_skills = root.count() - 1
    mapping = root.get_mapping()
    print("============================\n", root.print())
    discriminator = Head(
        input_dim=2, hid_dim=32, output_dim=num_skills, if_prob=True
    ).to(device)
    discrim_optimizer = optim.AdamW(
        discriminator.parameters(),
    )
    for n in nodes:
        n.value.env.discriminator = discriminator
        n.value.env.mapping = mapping
    # prepare data and mapping
    for _ in tqdm(range(iterations)):
        print("Sampling diffuse part")
        for n in nodes:
            state_buffer[n.key] = sample_diffuse(n, H, T, 10 * H)
        dataset = DiscriminatorDataset(state_buffer, mapping)
        weight = torch.ones(len(mapping.keys()))
        for node in nodes:
            weight[mapping[node.key]] = 0.25
        print("Training discriminator")
        train_discriminator(
            discriminator,
            discrim_optimizer,
            dataset,
            discrim_epoch,
            discrim_bs,
            weight,
        )
        min_discrim = np.inf
        min_key = np.inf
        for node in nodes:
            node_discrim = compute_discrim(discriminator, torch.tensor(np.array(state_buffer[node.key])).float(), mapping[node.key])
            if node_discrim < min_discrim:
                min_discrim = node_discrim
                min_key = node.key
        if min_discrim > discrim_threhold:
            return True, min_key, min_discrim
        # train policy
        # TODO: initialize with warm start
        print("Training policy") 
        for _ in range(p2d_ratio):
            for node in nodes:
                node.value.env.reset()
                sample_policy_init_state(node, T)
                node.value.learn(
                    total_timesteps=node.value.num_timesteps + T + H
                )  # reset_num_timesteps=True)
    return False, min_key, min_discrim


def create_node(z_max, parent, state_buffer, H, T, device):
    env = gym.make(
        "gym_examples/GridWorld-v0",
        size=10,
        render_mode="rgb_array",
    )
    env = gym_examples.UpsideWrapper(
        env,
        skill=z_max,
        discriminator=None,
        mapping=None,
        reward_start=T,
        reward_end=T+H,
    )
    env = gym_examples.AgentLocation(env)
    node = Node(
        key=z_max,
        value=DQN(
            "MlpPolicy",
            env,
            gradient_steps=5,
            train_freq=H + T,
            learning_starts=12,
            device=device,
            learning_rate=1e-2,
        ),
        parent=parent,
    )
    parent.add(node)
    state_buffer[z_max] = []


if __name__ == "__main__":
    # parameters
    discrim_threhold = 0.8  # (0, 1)
    n_start = 2
    n_end = 4
    z_max = 0
    num_skills = 0
    H = 2
    T = 2

    # init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Node(key=0, value=None, parent=None)
    queue = deque([root])
    state_buffer = {}

    while len(queue):
        parent = queue.popleft()
        print(f"Popping node {parent.key}")
        N = n_start
        for i in range(N):
            z_max += 1  # the key keeps incrementing, ensure that the keys of the upper level are larger than the lower level.
            # TODO: pointer and model should be updated
            # TODO: check DQN
            # TODO: add wrappers for non-termination but with truncation and reward model
            # DQN parameters: train_freq, learning_starts, tau, batch_size, lr, gamma
            create_node(z_max, parent, state_buffer, H, T, device)

        success, min_key, min_discrim = policy_learning(
            state_buffer, root, parent.children, discrim_threhold, H, T, device
        )
        print(success, min_key, min_discrim)
        if success:
            while success and N < n_end:
                N += 1
                z_max += 1
                create_node(z_max, parent, state_buffer, H, T, device)
                success, min_key, min_discrim = policy_learning(
                    state_buffer,
                    root,
                    parent.children,
                    discrim_threhold,
                    H,
                    T,
                    device,
                )
                print("== ", success, min_key, min_discrim)
        else:
            while not success and N > 1:
                N -= 1
                root.remove(min_key)
                success, min_key, min_discrim = policy_learning(
                    state_buffer,
                    root,
                    parent.children,
                    discrim_threhold,
                    H,
                    T,
                    device,
                )
                print("-- ", success, min_key, min_discrim)
        for node in parent.children:
            queue.append(node)

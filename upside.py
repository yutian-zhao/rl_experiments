from utils import *
from models import *
import gym_examples
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
import pickle
import math


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


class DQN:
    def __init__(
        self,
        obs_dim,
        n_actions,
        hid_dim,
        device,
        total_steps,
        warm_start_steps=None,
        target_update_freq=10,
        gamma=0.95,
        lr=1e-3,
        batchsize=32,
        buffer_size=100,
        tau=None,
        eps_start=0.9,
        eps_end=0,
        eps_min=0.05,
    ):
        self.batchsize = batchsize
        self.hid_dim = hid_dim
        self.device = device
        self.tau = tau  # 0.005
        self.target_update_freq = target_update_freq
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_min = eps_min
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayMemory(self.buffer_size)
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.q_network = Head(
            self.obs_dim,
            self.n_actions,
            hid_dim=self.hid_dim,
        ).to(device)
        self.target_net = Head(
            self.obs_dim,
            self.n_actions,
            hid_dim=self.hid_dim,
        ).to(device)
        self.target_net.eval()
        self.target_net.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.AdamW(self.q_network.parameters(), self.lr)
        self.steps_done = 0
        self.total_steps = total_steps
        self.warm_start_steps = warm_start_steps

    def select_action(self, state, eps_threshold=None):
        if eps_threshold is None:
            eps_threshold = epsilon(
                self.eps_start,
                self.eps_end,
                self.steps_done,
                self.total_steps,
                self.eps_min,
            )
            tqdm.write(f"Select action: eps_threshold={eps_threshold}")

        if random.random() > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # [B, S] -> [B, A]
                if type(state) != torch.Tensor:
                    state = torch.Tensor(state).to(self.device)
                q_values = self.q_network(state)
                return q_values.max(0).indices.item()
        else:
            # ASSUME: action index from 0
            return random.sample(range(self.n_actions), 1)[
                0
            ]  # torch.tensor(random.sample(range(self.n_actions), 1), device=device,) # dtype=torch.long

    def update(self):
        # warm_start_steps
        if self.warm_start_steps:
            if len(self.replay_buffer) < self.warm_start_steps:
                return
        transitions = self.replay_buffer.sample(
            min(len(self.replay_buffer), self.batchsize)
        )
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.q_network(state_batch)[
            torch.arange(min(len(self.replay_buffer), self.batchsize)),
            action_batch.squeeze().int(),
        ]  # [B]

        next_state_values = torch.zeros(
            (min(len(self.replay_buffer), self.batchsize)), device=device
        )
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma
        ) + reward_batch
        # print(next_state_values.shape, state_action_values.shape, expected_state_action_values.shape, reward_batch.shape)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        td_error = criterion(state_action_values, expected_state_action_values)
        # print(next_state_values * GAMMA, reward_batch.unsqueeze(1))
        # print(td_error)

        # Optimize the model
        self.optimizer.zero_grad()
        td_error.backward()
        # In-place gradient clipping
        nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()

        return td_error

    def update_target_network(self):
        if self.tau:
            for key in self.target_net.state_dict().keys():
                self.target_net.state_dict()[
                    key
                ] = self.q_network.state_dict()[
                    key
                ] * self.tau + self.target_net.state_dict()[
                    key
                ] * (
                    1 - self.tau
                )
            self.target_net.load_state_dict(self.target_net.state_dict())
        else:
            self.target_net.state_dict().load_state_dict(
                self.q_network.state_dict()
            )


class DiscriminatorDataset(Dataset):
    def __init__(self, buffer, mapping):
        self.data = set()
        self.mapping = mapping
        for key, value in buffer.items():
            for v in value:
                self.data.add((tuple(v), mapping[key]))
        self.data = list(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor([*self.data[idx][0]]), self.data[idx][1]


def rollout(env, policy, steps, eps_threshold=None):
    # return obs after n steps
    obs = env.unwrapped.obs
    for _ in range(steps):
        action = policy.select_action(obs, eps_threshold)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            obs = env.reset()
    return obs


def serial_rollout(env, policies, T, H, eps_threshold=None):
    # return obs after (n+1)*steps for n policies
    for policy in policies:
        rollout(env, policy, T, eps_threshold)
    return rollout(env, policies[-1], H)


def policy_only_serial_rollout(env, policies, T, eps_threshold=None):
    # return obs after (n+1)*steps for n policies
    obs = None
    for policy in policies:
        obs = rollout(env, policy, T, eps_threshold)
    return obs


def sample_diffuse(env, node, T, H, num, eps_threshold=None):
    # sample num states in the diffuse part of a node
    states = []
    for _ in range(num//H):
        env.reset()
        s = sample_policy_init_state(
            env, node, T, ignore_last=False, eps_threshold=eps_threshold
        )
        for _ in range(H):
            action = env.action_space.sample()
            state, _, terminated, truncated, info = env.step(action)  # vec_env
            # if terminated or truncated:
            # TODO: handle vec_env
            # state = info[0]['terminal_observation']
            if terminated:
                # ignore truncated
                state, _ = env.reset()
            states.append(state)
    return states


def sample_policy_init_state(
    env, node, T, ignore_last=False, eps_threshold=None
):
    skills = []
    node_pointer = node
    while node_pointer.parent:
        skills.append(node_pointer.value)
        node_pointer = node_pointer.parent
    # skills = skills[:-1] # remove root
    if ignore_last:
        skills = skills[:-1:-1]
    else:
        skills = skills[::-1]
    state = policy_only_serial_rollout(env, skills, T, eps_threshold)
    return state


def collect_rollout(env, node, T, H=None, eps_threshold=None):
    rollouts = []
    obs = env.unwrapped.obs
    for _ in range(T):
        action = node.value.select_action(obs, eps_threshold)
        next_obs, _, terminated, truncated, info = env.step(action)
        rollouts.append([obs, action, next_obs, 0])
        if terminated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    if H:
        for _ in range(H):
            action = env.action_space.sample()
            next_obs, _, terminated, truncated, info = env.step(action)
            rollouts.append([obs, action, next_obs, 0])
            if terminated:
                obs, _ = env.reset()
            else:
                obs = next_obs
    return rollouts


def plot_trajectories(
    env, root, mapping, T, H, fname_prefix, num_eval_episodes=10, options=None
):
    size = env.unwrapped.size
    # plt.xlim(0, env.unwrapped.size)
    # plt.ylim(0, env.unwrapped.size)
    num_skills = len(mapping.keys())
    visit_map = np.zeros((size, size, num_skills))
    n_col = 4
    n_row = math.ceil(num_skills / n_col)
    plt.figure()
    plt.axis((0, size, 0, size))
    obs, _ = env.reset(options=options)
    plt.plot(obs[0], obs[1], "*", c="black")
    plt.grid(which='both')

    for eps in range(num_eval_episodes):
        for i, skill in enumerate(mapping.keys()):
            node = root.find(skill)
            obs, _ = env.reset(options=options)
            # tqdm.write(str(obs))
            skill_seq = []
            node_pointer = node
            while node_pointer.parent:
                skill_seq.append(node_pointer)
                node_pointer = node_pointer.parent
            skill_seq = skill_seq[::-1]
            rollouts = []
            for s in skill_seq:
                if eps == 0:
                    rollouts += collect_rollout(env, s, T, eps_threshold=0)
                else:
                    rollouts += collect_rollout(env, s, T, eps_threshold=0.5)
            trajectory = list(Transition(*zip(*rollouts)).state) + [
                Transition(*zip(*rollouts)).next_state[-1]
            ]
            trajectory = np.array(list(zip(*trajectory)), dtype=np.float64)
            for j in range(len(trajectory[0])):
                visit_map[int(trajectory[0][j]), int(trajectory[1][j]), int(mapping[skill])] += 1

            if eps == 0:
                trajectory += np.random.uniform(-0.2, 0.2, trajectory.shape)
                # vertical x, horizonal y
                plt.plot(
                    trajectory[1],
                    trajectory[0],
                    "-o",
                    c=cm.gist_rainbow(mapping[skill] / len(mapping.keys())),
                    label=skill,
                )

    # plt.legend()
    plt.savefig(fname_prefix + f"_{datetime.now().strftime('%m%d%H%M')}.jpg")
    plt.close()

    plt.figure(figsize=(20, 16))
    for i in range(num_skills):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(visit_map[:, :, i] + 0.01, cmap="Greens", origin='lower')
        for (ii, j), z in np.ndenumerate(visit_map[:, :, i]):
            plt.text(j, ii, '{}'.format(int(z)), ha='center', va='center')
    plt.savefig(
        fname_prefix + f"_visit_{datetime.now().strftime('%m%d%H%M')}.jpg"
    )
    plt.close()


def visualize_discrim_subplot(
    env,
    discrim,
    num_skills,
    device,
    fname_prefix,
):
    size = env.unwrapped.size
    n_col = 4
    n_row = math.ceil(num_skills / n_col)
    states = torch.tensor(np.concatenate(
        np.transpose(
            np.dstack((np.meshgrid(range(size), range(size)))), (1, 0, 2)
        )
    ), dtype=torch.float32, device=device)
    with torch.no_grad():
        output = discrim(states)
    for i in range(num_skills):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(output[:, i].cpu().numpy().reshape(size, size), origin='lower')
    plt.savefig(fname_prefix + f"_{datetime.now().strftime('%m%d%H%M')}.jpg")
    plt.close()


def visualize_qtable_subplot(
    env,
    root,
    mapping,
    device,
    fname_prefix,
):
    num_skills = len(mapping.keys())
    size = env.unwrapped.size
    n_col = 4
    n_row = math.ceil(num_skills / n_col)
    states = torch.tensor(np.concatenate(
        np.transpose(
            np.dstack((np.meshgrid(range(size), range(size)))), (1, 0, 2)
        )
    ), dtype=torch.float32, device=device)

    for key in mapping.keys():
        node = root.find(key)
        with torch.no_grad():
            output = node.value.q_network(states)
        plt.subplot(n_row, n_col, mapping[key] + 1)
        plt.imshow(output.max(axis=1).values.cpu().numpy().reshape(size, size), cmap="Purples", origin='lower')

    plt.savefig(fname_prefix + f"_{datetime.now().strftime('%m%d%H%M')}.jpg")
    plt.close()


# def visit_map(env, skill, T, num_eval_episodes=10, options=None):
#     size = env.unwrapped.size
#     map = np.zeros((size, size))
#     for eps in range(num_eval_episodes):
#         obs, _ = env.reset(options=options)
#         map[obs[0]][obs[1]] += 1
#         rollouts = collect_rollout(env, skill, T, eps_threshold=0.2)
#         for obs, action, next_obs, reward in rollouts:
#             map[next_obs[0]][next_obs[1]] += 1
#     return map


def visualize_discrim(
    env,
    discrim,
    num_skills,
    device,
    fname_prefix,
):
    size = env.unwrapped.size
    map = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            state = torch.tensor([i, j], dtype=torch.float32, device=device)
            map[i][j] = discrim(state).argmax().item() / num_skills
    plt.imshow(map, cmap="gist_rainbow", origin='lower')
    plt.savefig(fname_prefix + f"_{datetime.now().strftime('%m%d%H%M')}.jpg")
    plt.close()


def plot_lists(lsts, labels=None, fname_prefix=None):
    if not labels:
        labels = [*range(len(lsts))]
    for i in range(len(lsts)):
        if type(lsts[i]) == torch.Tensor or type(lsts[i][0]) == torch.Tensor:
            lsts[i] = torch.tensor(lsts[i], device="cpu")
        plt.plot([*range(len(lsts[i]))], lsts[i], label=labels[i])
    plt.legend()
    if fname_prefix:
        plt.savefig(
            fname_prefix + f"_{datetime.now().strftime('%m%d%H%M')}.jpg"
        )
    plt.close()


def train_discriminator(
    discriminator, optimizer, dataset, epochs, bs, weight=None
):
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [0.8, 0.2]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=bs, shuffle=True)
    stopper = EarlyStopper(patience=100, min_delta=0.01, if_save=True)
    training_losses = []
    valid_losses = []

    for epoch in range(epochs):
        num_batch = 0
        train_loss_sum = 0
        discriminator.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            X = X.float()
            # Compute prediction error
            pred = discriminator(X)
            # print(pred.shape, y.shape)
            loss = torch.nn.functional.cross_entropy(
                pred, y, weight=weight.to(device)
            )
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
            loss = torch.nn.functional.cross_entropy(
                pred, y
            )  # , weight=weight
            num_batch += 1
            valid_loss_sum += loss

        valid_loss = valid_loss_sum / num_batch
        valid_losses.append(valid_loss)

        if stopper.early_stop(
            valid_loss,
            state_dict=discriminator.state_dict(),
            path="upside_model/upside_discrim_temp.dict",
        ):
            discriminator.load_state_dict(
                torch.load("upside_model/upside_discrim_temp.dict")
            )
            break

        if epoch % 100 == 0:
            tqdm.write(f"train loss: {train_loss_sum / num_batch}")
            tqdm.write(f"valid loss: {valid_loss}")

    return training_losses, valid_losses


def compute_discrim(discriminator, input, target):
    discriminator.eval()
    return torch.mean(
        F.softmax(discriminator(input), dim=1)[:, target]
    )  # [B, C] -> [B] -> [1]


def policy_learning(
    env,
    discriminator,
    discrim_optimizer,
    state_buffer,
    root,
    nodes,
    discrim_threhold,
    T,
    H,
    device,
    prefix,
):
    # parameters
    step_limit = 100000
    p2d_ratio = 1
    discrim_epoch = 1
    iterations = 10000
    discrim_bs = 256
    k_policy = 1

    mapping = root.get_mapping()
    num_skills = len(mapping.keys())
    size = env.unwrapped.size
    n_col = 4
    n_row = math.ceil(num_skills / n_col)
    tqdm.write("============================\n" + root.print())

    # env.discriminator = discriminator
    # env.mapping = mapping
    # prepare data and mapping
    for i in tqdm(range(iterations)):
        # tqdm.write("Sampling diffuse part")
        for n in nodes:
            env.reset()
            state_buffer[n.key] = sample_diffuse(
                env, n, T, H, 10 * H, eps_threshold=0
            )
        dataset = DiscriminatorDataset(state_buffer, mapping)
        
        weight = torch.ones(len(mapping.keys()))
        for node in nodes:
            weight[mapping[node.key]] = 0.25
        # tqdm.write("Training discriminator")
        training_losses, valid_losses = train_discriminator(
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
            node_discrim = compute_discrim(
                discriminator,
                torch.tensor(np.array(state_buffer[node.key]))
                .float()
                .to(device),
                mapping[node.key],
            )
            if node_discrim < min_discrim:
                min_discrim = node_discrim
                min_key = node.key
        # at leart trian policy for one time
        if min_discrim > discrim_threhold and i > 0:
            return True, min_key, min_discrim
        # train policy
        # TODO: initialize with warm start
        tqdm.write(f"Training policy min_discrim: {min_discrim}\n")
        rewards = [[] for _ in range(len(nodes))]
        td_error_lst = [[] for _ in range(len(nodes))]
        for _ in tqdm(range(p2d_ratio)):
            for ni, node in enumerate(nodes):
                reward, td_errors = train_policy(
                    env, discriminator, node, T, H, k_policy, mapping, device
                )
                rewards[ni].append(reward)
                td_error_lst[ni] += td_errors
        # plot
        if i%100==99:
        # plot_lists(
        #     [training_losses, valid_losses],
        #     ["train", "valid"],
        #     f"upside_plot/{prefix}_train_discrim_i{i}",
        # )
        # plot_lists(
        #     rewards,
        #     [n.key for n in nodes],
        #     f"upside_plot/{prefix}_train_policy_i{i}",
        # )
        # plot_lists(
        #     td_error_lst,
        #     [n.key for n in nodes],
        #     f"upside_plot/{prefix}_td_i{i}",
        # )
            plot_trajectories(
                env, root, mapping, T, H, f"upside_plot/{prefix}_trac_i{i}"
            )
            visualize_discrim(
                env,
                discriminator,
                len(mapping.keys()),
                device,
                f"upside_plot/{prefix}_disc_{i}",
            )
            visualize_discrim_subplot(
                env,
                discriminator,
                len(mapping.keys()),
                device,
                f"upside_plot/{prefix}_disc_subplot_{i}",
            )
            visualize_qtable_subplot(
                env, root, mapping, device, f"upside_plot/{prefix}_qtable_{i}"
            )

            diffuse_map = np.zeros((size, size, num_skills))
            for x, y in dataset:
                diffuse_map[x[0],x[1],y] +=  1
            plt.figure(figsize=(20, 16))
            for ii in range(num_skills):
                plt.subplot(n_row, n_col, ii + 1)
                plt.imshow(diffuse_map[:, :, ii] + 0.01, cmap="Blues", origin='lower')
                for (ii, j), z in np.ndenumerate(diffuse_map[:, :, ii]):
                    plt.text(j, i, '{}'.format(int(z)), ha='center', va='center')
            plt.savefig(
                f"upside_plot/{prefix}_diffuse_{i}_{datetime.now().strftime('%m%d%H%M')}.jpg"
            )
            plt.close()
    return False, min_key, min_discrim


def train_policy(env, discriminator, node, T, H, k_policy, mapping, device):
    # print(f"Training policy for node {node.key}"+"\n")
    env.reset()
    sample_policy_init_state(
        env, node, T, ignore_last=True, eps_threshold=0.5
    )  # stochasity
    rollouts = collect_rollout(env, node, T, H, eps_threshold=0.5)
    sum_reward = 0
    td_errors = []
    for rollout in rollouts[T:]:
        r = F.log_softmax(
            discriminator(
                torch.tensor(rollout[-2], device=device, dtype=torch.float)
            ),
            dim=0,
        )[mapping[node.key]].item()
        # rollout[-1] = r
        sum_reward += r
    rollouts[T - 1][-1] = sum_reward
    for obs, action, next_obs, reward in rollouts[:T]:
        node.value.replay_buffer.push(
            torch.tensor(obs, dtype=torch.float, device=device).reshape(
                (1, -1)
            ),
            torch.tensor([action], dtype=torch.float, device=device),
            torch.tensor(next_obs, dtype=torch.float, device=device).reshape(
                (1, -1)
            ),
            torch.tensor([reward], dtype=torch.float, device=device),
        )
    node.value.steps_done += T + H
    for _ in range(k_policy):
        td_error = node.value.update()
        td_errors.append(td_error)
    return sum_reward, td_errors


def create_node(z_max, dqn, parent, state_buffer, T, H, device):

    node = Node(
        key=z_max,
        value=dqn,
        parent=parent,
    )
    parent.add(node)
    state_buffer[z_max] = []


if __name__ == "__main__":

    # T, H = 10, 10
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # root = Node(key=0, value=None, parent=None)
    # create_node(1, root, {}, T, H, device=device)
    # train_policy(root.children[0],T, H,)
    # print("counter: ", root.children[0].value.env.envs[0].counter)
    # print("dqn counter: ", root.children[0].value.counter)

    # parameters
    discrim_threhold = 0.7  # (0, 1)
    n_start = 2
    n_end = 4
    z_max = 0
    H = 2
    T = 4
    discrim_hid = 16
    dqn_hid = 16
    total_steps = 1000
    discrim_lr = 1e-3
    dqn_buffer_size = T # 8 * T
    dqn_batchsize = 256

    # exit(0)

    # init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Node(key=0, value=None, parent=None)
    queue = deque([root])
    state_buffer = {}
    env = gym.make(
        "gym_examples/GridWorld-v0",
        size=10,
        render_mode="rgb_array",
    )
    env = gym_examples.UpsideWrapper(env)
    env = gym_examples.AgentLocation(env)
    prefix = datetime.now().strftime("%m%d%H%M")
    loop_counter = 0

    while len(queue):
        parent = queue.popleft()
        tqdm.write(f"Popping node {parent.key}")
        N = n_start
        loop_counter += 1
        for i in range(N):
            z_max += 1  # the key keeps incrementing, ensure that the keys of the upper level are larger than the lower level.
            # TODO: pointer and model should be updated
            # TODO: check DQN
            # TODO: add wrappers for non-termination but with truncation and reward model
            # TODO: DQN parameters: train_freq, learning_starts, tau, batch_size, lr, gamma
            dqn = DQN(
                obs_dim=env.observation_space.shape[0],
                n_actions=env.action_space.n,
                hid_dim=dqn_hid,
                device=device,
                total_steps=total_steps,
                buffer_size=dqn_buffer_size,
                batchsize=dqn_batchsize,
            )
            create_node(z_max, dqn, parent, state_buffer, T, H, device)

        discriminator = Head(
            input_dim=env.observation_space.shape[0],
            hid_dim=discrim_hid,
            output_dim=root.count() - 1,
            if_softmax=False,
        ).to(device)
        discrim_optimizer = optim.AdamW(discriminator.parameters(), discrim_lr)
        success, min_key, min_discrim = policy_learning(
            env,
            discriminator,
            discrim_optimizer,
            state_buffer,
            root,
            parent.children,
            discrim_threhold,
            T,
            H,
            device,
            f"{prefix}_o{loop_counter}_c0",
        ) # 
        print(f"{success}, {min_key}, {min_discrim}")
        if success:
            while success and N < n_end:
                N += 1
                z_max += 1
                dqn = DQN(
                    obs_dim=env.observation_space.shape[0],
                    n_actions=env.action_space.n,
                    hid_dim=dqn_hid,
                    device=device,
                    total_steps=total_steps,
                )
                create_node(z_max, dqn, parent, state_buffer, T, H, device)
                discriminator = Head(
                    input_dim=env.observation_space.shape[0],
                    hid_dim=discrim_hid,
                    output_dim=root.count() - 1,
                    if_softmax=False,
                ).to(device)
                discrim_optimizer = optim.AdamW(
                    discriminator.parameters(), discrim_lr
                )
                success, min_key, min_discrim = policy_learning(
                    env,
                    discriminator,
                    discrim_optimizer,
                    state_buffer,
                    root,
                    parent.children,
                    discrim_threhold,
                    T,
                    H,
                    device,
                    f"{prefix}_o{loop_counter}_c1",
                ) # 
                print(f"== , {success}, {min_key}, {min_discrim}")
                # TODO
                if not success:
                    N -= 1
                    root.remove(min_key)
                    state_buffer.pop(min_key)
        else:
            while not success and len(parent.children):  # N>1
                N -= 1
                root.remove(min_key)
                state_buffer.pop(min_key)
                if len(parent.children) > 0:
                    discriminator = Head(
                        input_dim=env.observation_space.shape[0],
                        hid_dim=discrim_hid,
                        output_dim=root.count() - 1,
                        if_softmax=False,
                    ).to(device)
                    discrim_optimizer = optim.AdamW(
                        discriminator.parameters(), discrim_lr
                    )
                    success, min_key, min_discrim = policy_learning(
                        env,
                        discriminator,
                        discrim_optimizer,
                        state_buffer,
                        root,
                        parent.children,
                        discrim_threhold,
                        T,
                        H,
                        device,
                        f"{prefix}_o{loop_counter}_c2",
                    ) # 
                    tqdm.write(f"-- {success}, {min_key}, {min_discrim}")
                else:
                    tqdm.write(root.print())
        for node in parent.children:
            queue.append(node)
    with open(
        f"upside_model/upside_model_{datetime.now().strftime('%m%d%H%M')}.pkl",
        "wb",
    ) as fout:
        pickle.dump(root, fout)
"""
0,|
0:1,2,3,4,|
1:5,6,|2:9,|3:10,12,|4:14,15,|
5:17,|6:18,|9:x|10:23,|12:x|14:x|15:29,|
17:x|18:x|23:x|29:36,|
36:x|
"""
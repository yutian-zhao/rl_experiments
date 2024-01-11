import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym

class FeaturesExtractor(nn.Module):
    def __init__(self, observation_space: gym.Space, features_dim: int = 128, normalized_image: bool = False, concat=1) -> None:
        # TODO: normalize
        super().__init__()
        # super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[-1]*concat
        self.features_dim = features_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 2, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(2, 4, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(4, 8, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()).unsqueeze(0).permute(0, 3, 1, 2).repeat(1, concat, 1, 1).float()).shape[1]  # [1,7,7,3]


        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    
class Head(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=None, if_prob=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dim = hid_dim if hid_dim else self.input_dim
        self.linear = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.output_dim),
        )
        self.if_prob = if_prob

    def forward(self, x, extra_x=None):
        # TODO: Relu
        # print("prev_x: ", x.size())
        if extra_x is not None:
            # TODO: change to one-hot encoding and may add Norm2d
            x = torch.cat((x, extra_x), dim=1)
            # print("extra_x: ", extra_x.size())
            # print("post_x: ", x.size())
        x = self.linear(x)
        if self.if_prob:
            return nn.functional.softmax(x, dim=1)
        else:
            return x
        

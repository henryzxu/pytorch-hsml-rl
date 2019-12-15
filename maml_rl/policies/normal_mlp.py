import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import OrderedDict
from maml_rl.policies.policy import Policy, weight_init

class NormalMLPPolicy(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a 
    `Normal` distribution output, with trainable standard deviation. This 
    policy network can be used on tasks with continuous action spaces (eg. 
    `HalfCheetahDir`). The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_gauss_mlp_policy.py
    """
    def __init__(self, input_size, output_size, hidden_sizes=(), tree=None,
                 nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
        super(NormalMLPPolicy, self).__init__(
            input_size=input_size, output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i),
                nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.mu = nn.Linear(layer_sizes[-1], output_size)

        self.tree = tree
        tree.to(torch.device("cuda"))

        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
        self.apply(weight_init)


    def forward(self, input, task=None, params=None, enhanced=False):
        if params is None:
            params = OrderedDict(self.named_parameters())


        # if self.env_name == 'AntPos-v0':
        _, embedding = self.tree(torch.from_numpy(task))
        # if self.env_name == 'AntVel-v1':
        #     _, embedding = self.tree(torch.from_numpy(np.array([task["velocity"]])))

        print(input.shape)
        if len(input.shape) == 2:
            output = torch.t(
                torch.stack([torch.cat([teo.clone(), embedding[0].clone()], 0) for teo in input], 1).clone())
        if len(input.shape) == 3:
            output = torch.stack([torch.t(
                torch.stack([torch.cat([teo.clone(), embedding[0].clone()], 0) for teo in tei], 1).clone()) for tei in input], 1).permute(1, 0, 2)


        # output = input
        print(output.shape)
        for i in range(1, self.num_layers):
            output = F.linear(output,
                weight=params['layer{0}.weight'.format(i)],
                bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        mu = F.linear(output, weight=params['mu.weight'],
            bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

        return Normal(loc=mu, scale=scale)

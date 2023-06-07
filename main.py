from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, set_exploration_mode
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

import argparse
import gym


def get_parameters():
    parser = argparse.ArgumentParser(description='Argument Parser Example')

    ## main hyperparams
    parser.add_argument('--use-cuda', action='store_true', default=False)
    parser.add_argument('--num_cells', type=int, default=256, help='Number of cells in each layer')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm')

    # environemnt hyperparams
    parser.add_argument('--env', type=str, choices=['MsPacman', 'Pong'], 
                                 default='Pong', help='Gym environment name')
    frame_skip = 1
    frames_per_batch = 1_000 // frame_skip
    # For a complete training, bring the number of frames up to 1M
    total_frames = 50_000 // frame_skip

    sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 10  # optimization steps per batch of data collected
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4

    args = parser.parse_args()

    # Check if the device argument is valid
    if args.use_cuda and not torch.cuda.is_available():
        parser.error("CUDA is not available. Please choose 'cpu' for the device.")

    args.device = 'cuda:0' if args.use_cuda else 'cpu'
    args.env = f'ALE/{args.env}-v5'
    del args.use_cuda

    return args

if __name__ == '__main__':
    args = get_parameters()
    base_env = gym.make(args.env)

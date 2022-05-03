import os
import typing
import logging
import traceback
import numpy as np
from collections import Counter
import time

import torch
from torch import multiprocessing as mp

from .env_utils import Environment
from douzero.env import Env
from douzero.env.env import _cards2array

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def create_env(flags):
    return Env(flags.objective)


def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    该函数将基于从完整队列中接收到的索引从缓冲区中采样一批数据。它还将释放索引，将其发送到full_queue。
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch


def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    Change the optimizer from RMSprop to SGD
    """
    positions = ['A', 'B', 'C', 'D', 'showA', 'showB', 'showC', 'showD']
    # positions = ['play', 'show']
    optimizers = {}
    for position in positions:
        # optimizer = torch.optim.SGD(
        #     learner_model.parameters(position),
        #     lr=flags.learning_rate,
        #     momentum=flags.momentum,
        #     # eps=flags.epsilon,
        #     # alpha=flags.alpha
        # )
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)
        optimizers[position] = optimizer
    return optimizers


def create_buffers(flags, device_iterator):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.

    319:
    430:

    """
    T = flags.unroll_length
    # positions = ['landlord', 'landlord_up', 'landlord_down']
    positions = ['A', 'B', 'C', 'D']
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            # x_dim = 319 if position == 'landlord' else 430
            x_dim = 728  # TODO modify the x_dim
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
                obs_action=dict(size=(T, 52), dtype=torch.int8),
                obs_z=dict(size=(T, 5, 156), dtype=torch.int8),
                show_obs_x_no_action=dict(size=(T, 260), dtype=torch.int8),
                show_obs_action=dict(size=(T, 52), dtype=torch.int8),
                show_target=dict(size=(T, ), dtype=torch.float32)
            )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:' + str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers


def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    positions = ['A', 'B', 'C', 'D']
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        # show card buffer
        show_obs_x_no_action_buf = {p: [] for p in positions}
        show_obs_action_buf = {p: [] for p in positions}
        show_target_buf = {p: [] for p in positions}

        position, obs, env_output = env.initial(model)

        all_show_obs = env_output['all_show_obs']

        while True:
            # add the show card logic
            for pos in ['A', 'B', 'C', 'D']:
                show_obs_x_no_action_buf[pos].append(
                    torch.from_numpy(all_show_obs[pos]['obs']['x_no_action']).to(device))
                # print((torch.from_numpy(all_show_obs[pos]['obs']['x_no_action']).to(device)))
                show_obs_action_buf[pos].append(_cards2tensor(all_show_obs[pos]['action']))
            while True:
                # not game over
                obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                obs_z_buf[position].append(env_output['obs_z'])
                with torch.no_grad():
                    agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags)
                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][_action_idx]
                obs_action_buf[position].append(_cards2tensor(action))
                # every loop
                size[position] += 1
                position, obs, env_output = env.step(action)
                # game over
                if env_output['done']:
                    all_show_obs = env_output['all_show_obs']
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff - 1)])
                            done_buf[p].append(True)
                            episode_return = env_output['episode_return'][p]
                            episode_return_buf[p].extend([0.0 for _ in range(diff - 1)])
                            episode_return_buf[p].append(episode_return)
                            target_buf[p].extend([episode_return for _ in range(diff)])
                            # show target
                            show_target_buf[p].append(episode_return)
                    break
            for p in positions:
                while size[p] > T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    cnt = 0
                    num = 0
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
                        buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
                        buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
                        if cnt % 13 == 0:
                            buffers[p]['show_obs_x_no_action'][index][num, ...] = show_obs_x_no_action_buf[p][num]
                            buffers[p]['show_obs_action'][index][num, ...] = show_obs_action_buf[p][num]
                            buffers[p]['show_target'][index][num, ...] = show_target_buf[p][num]
                            num += 1
                        cnt += 1
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    show_obs_x_no_action_buf[p] = show_obs_x_no_action_buf[p][num:]
                    show_obs_action_buf[p] = show_obs_action_buf[p][num:]
                    show_target_buf[p] = show_target_buf[p][num:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p] = obs_action_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    size[p] -= T
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e


def _cards2tensor(list_cards):
    """
    Convert a list of integers to the tensor
    representation
    See Figure 2 in https://arxiv.org/pdf/2106.06135.pdf
    """
    matrix = _cards2array(list_cards)
    matrix = torch.from_numpy(matrix)
    return matrix

"""
Here, we wrap the original environment to make it easier
to use. When a game is finished, instead of mannualy reseting
the environment, we do it automatically.
"""
import numpy as np
import torch 

def _format_observation(obs, device):
    """
    一个实用函数处理观察，并将它们移动到CUDA。
    A utility function to process observations and
    move them to CUDA.
    """
    position = obs['position']
    if not device == "cpu":
        device = 'cuda:' + str(device)
    device = torch.device(device)
    x_batch = torch.from_numpy(obs['x_batch']).to(device)
    z_batch = torch.from_numpy(obs['z_batch']).to(device)
    x_no_action = torch.from_numpy(obs['x_no_action'])
    z = torch.from_numpy(obs['z'])
    obs = {'x_batch': x_batch,
           'z_batch': z_batch,
           'legal_actions': obs['legal_actions'],
           }
    return position, obs, x_no_action, z


def _format_show_observation(obs, device):
    if not device == "cpu":
        device = 'cuda:' + str(device)
    device = torch.device(device)
    x_batch = torch.from_numpy(obs['x_batch']).to(device)
    x_no_action = torch.from_numpy(obs['x_no_action'])
    obs = {'show_x_batch': x_batch,
           'legal_actions': obs['legal_actions'],
           'action': obs['action']
           }
    return obs, x_no_action

"""
self.env = Env()
"""
class Environment:
    def __init__(self, env, device):
        """
        trains环境包装类
        Initialzie this environment wrapper
        """
        self.env = env
        self.device = device
        self.episode_return = None
        self.model = None

    def initial(self, model):
        obs, all_show_obs = self.env.reset(model, self.device)
        initial_position, initial_obs, x_no_action, z = _format_observation(obs, self.device)
        self.model = model
        self.episode_return = {"A": torch.zeros(1, 1),
                               "B": torch.zeros(1, 1),
                               "C": torch.zeros(1, 1),
                               "D": torch.zeros(1, 1)}
        initial_done = torch.ones(1, 1, dtype=torch.bool)
        return initial_position, initial_obs, dict(
            done=initial_done,
            episode_return=self.episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
            all_show_obs=all_show_obs
            )
        
    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        for position in ['A', 'B', 'C', 'D']:
            self.episode_return[position] += reward[position]
        episode_return = self.episode_return
        all_show_obs=dict()
        # get the episode_return though the position
        if done:
            obs, all_show_obs = self.env.reset(self.model, self.device)
            self.episode_return = {"A": torch.zeros(1, 1),
                                   "B": torch.zeros(1, 1),
                                   "C": torch.zeros(1, 1),
                                   "D": torch.zeros(1, 1)}

        position, obs, x_no_action, z = _format_observation(obs, self.device)
        done = torch.tensor(done).view(1, 1)
        return position, obs, dict(
            done=done,
            episode_return=episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
            all_show_obs=all_show_obs
            )

    def close(self):
        self.env.close()

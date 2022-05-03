"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""
import traceback

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F



# Model dict is only used in evaluation but not training
model_dict = {}


class ShowCardModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(312, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        x = self.dense1(x)
        # x = nn.ReLU(x)
        x = F.leaky_relu(x, inplace=False)
        x = self.dense2(x)
        x = F.leaky_relu(x, inplace=False)
        # x = nn.ReLU(x)
        x = self.dense3(x)
        x = F.leaky_relu(x, inplace=False)
        # x = nn.ReLU(x)
        x = self.dense4(x)
        # x = nn.ReLU(x)
        x = F.leaky_relu(x, inplace=False)
        x = self.dense5(x)
        # x = nn.ReLU(x)
        x = F.leaky_relu(x, inplace=False)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action, max_value=torch.max(x))


class GModel(nn.Module):

    def __init__(self):
        super().__init__()
        """
        除了历史move外state加action
        """
        # TODO MODIFY SIZE 162->156
        """
        self.lstm = nn.LSTM(156, 128, batch_first=True)
        self.dense1 = nn.Linear(492, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)
        """
        self.lstm = nn.LSTM(156, 128, batch_first=True)
        self.dense1 = nn.Linear(908, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.dense3 = nn.Linear(1024, 1024)
        self.dense4 = nn.Linear(1024, 1024)
        self.dense5 = nn.Linear(1024, 1024)
        self.dense6 = nn.Linear(1024, 1)

    def forward(self, z, x, return_value=False, flags=None):
        """
        x: obs[x_batch]: other move
        z: obs[z_batch]: historical move
        """
        """
        x shape: torch.Size([1, 484])
        z shape: torch.Size([1, 5, 162])
        """
        """
        why have 280???
        """
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x, dim=0)[0]
            return dict(action=action)

model_dict['A'] = GModel
model_dict['B'] = GModel
model_dict['C'] = GModel
model_dict['D'] = GModel
model_dict['showA'] = ShowCardModel
model_dict['showB'] = ShowCardModel
model_dict['showC'] = ShowCardModel
model_dict['showD'] = ShowCardModel

class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """

    def __init__(self, device=0):

        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        # 将模型加载到device上
        self.models['A'] = GModel().to(torch.device(device))
        self.models['B'] = GModel().to(torch.device(device))
        self.models['C'] = GModel().to(torch.device(device))
        self.models['D'] = GModel().to(torch.device(device))
        # self.models['play'] = GModel().to(torch.device(device))
        self.models['showA'] = ShowCardModel().to(torch.device(device))
        self.models['showB'] = ShowCardModel().to(torch.device(device))
        self.models['showC'] = ShowCardModel().to(torch.device(device))
        self.models['showD'] = ShowCardModel().to(torch.device(device))

    def forward(self, position, z, x, training=False, flags=None):
        # model = None
        model = self.models[position]
        # if position in ['A', 'B', 'C', 'D']:
        #     model = self.models['play']
        # elif position == 'show':
        #     model = self.models['show']
        return model.forward(z, x, training, flags)

    def share_memory(self):
        # self.models['play'].share_memory()
        # self.models['show'].share_memory()
        self.models['A'].share_memory()
        self.models['B'].share_memory()
        self.models['C'].share_memory()
        self.models['D'].share_memory()
        self.models['showA'].share_memory()
        self.models['showB'].share_memory()
        self.models['showC'].share_memory()
        self.models['showD'].share_memory()

    def eval(self):
        # self.models['play'].eval()
        # self.models['show'].eval()
        self.models['A'].eval()
        self.models['B'].eval()
        self.models['C'].eval()
        self.models['D'].eval()
        self.models['showA'].eval()
        self.models['showB'].eval()
        self.models['showC'].eval()
        self.models['showD'].eval()

    def parameters(self, position):
        # if position in ['A', 'B', 'C', 'D']:
        #     return self.models['play'].parameters()
        # elif position == 'show':
        #     return self.models['show'].parameters()
        # raise Exception("position not found")
        return self.models[position].parameters()

    def get_model(self, position):
        # if position in ['A', 'B', 'C', 'D']:
        #     return self.models['play']
        # elif position == 'show':
        #     return self.models['show']
        # raise Exception("position not found")
        return self.models[position]

    def get_models(self):
        return self.models

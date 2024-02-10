import torch
from torch import nn

class Linear(nn.Module):
    """
    Simple linear network.
    """

    def __init__(self, num_features, num_timesteps, num_output=[4, 7]):
        
        super(Linear, self).__init__()

        self.layers = []

        self.layers.append(nn.Flatten())

        self.fc0 = nn.Linear(num_features*num_timesteps, num_output[0])
        self.fc1 = nn.Linear(num_features*num_timesteps, num_output[1])
        # nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('sigmoid'))
        # self.fc1.bias.data.fill_(0.01) # NOTE: Gewoon gevonden online!!!

        self.act_fn = nn.Softmax(dim=-1)
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):

        out = self.layers(x)
        out_avpu = self.act_fn(self.fc0(out))
        out_crt = self.act_fn(self.fc1(out))

        return out_avpu, out_crt
    
    def device(self):
        """ Return the device the model is currently on. """
        return next(self.parameters()).device

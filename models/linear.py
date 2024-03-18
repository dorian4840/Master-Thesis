import torch
from torch import nn

class Linear(nn.Module):
    """
    Simple linear network.
    """

    def __init__(self, num_features, num_timesteps, hidden_dims, act_fn='relu',
                 num_output=[4, 7], linear_dropout=0):
        
        super().__init__()

        self.layers = []
        self.layers.append(nn.Flatten())

        input_dims = num_features*num_timesteps
        for hidden in hidden_dims:

            l = nn.Linear(input_dims, hidden)
            nn.init.xavier_uniform_(l.weight, gain=nn.init.calculate_gain('sigmoid'))
            l.bias.data.fill_(0.01)
            self.layers.append(l)

            self.layers.append(nn.Dropout(p=linear_dropout))

            if act_fn == 'relu':
                self.layers.append(nn.ReLU())
            elif act_fn == 'leaky_relu':
                self.layers.append(nn.LeakyReLU())
            elif act_fn == 'gelu':
                self.layers.append(nn.GELU())
            elif act_fn == 'tanh':
                self.layers.append(nn.Tanh())
            else:
                raise NotImplementedError
            
            input_dims = hidden


        self.layers = nn.Sequential(*self.layers)

        self.fc0 = nn.Linear(hidden, num_output[0])
        nn.init.xavier_uniform_(self.fc0.weight, gain=nn.init.calculate_gain('sigmoid'))
        self.fc0.bias.data.fill_(0.01)

        self.fc1 = nn.Linear(hidden, num_output[1])
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('sigmoid'))
        self.fc1.bias.data.fill_(0.01)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        out = self.layers(x)
        out_avpu = self.softmax(self.fc0(out))
        out_crt = self.softmax(self.fc1(out))

        return out_avpu, out_crt


    def device(self):
        """ Return the device the model is currently on. """
        return next(self.parameters()).device

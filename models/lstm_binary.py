import torch
from torch import nn


class LSTM(nn.Module):
    """
    Use the pytorch_tcn TCN and add a linear layer at the end to reduce the
    output to a desired length.
    """

    def __init__(self, name, num_input, hidden_size, num_layers, num_output=2,
                 dropout=0, final_activation='sigmoid'):

        super(LSTM, self).__init__()

        self.name = name
        self.num_input = num_input
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_output = num_output
        self.dropout = dropout
        self.final_activation = final_activation

        # LSTM
        self.lstm = nn.LSTM(num_input, hidden_size, num_layers,
                             dropout=dropout, batch_first=True)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, num_output)
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('sigmoid'))
        self.fc.bias.data.fill_(0.01)

        if final_activation == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif final_activation == 'softmax':
            self.act_func = nn.Softmax(dim=-1)
        else:
            raise ValueError(f'{final_activation} not a valid final activation function')


    def forward(self, x):
        """
        Forward data through the model.
        TODO: Make it ignore missing data.
        """

        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(self.device())
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).requires_grad_().to(self.device())

        self.lstm.flatten_parameters()
        output, (hn, cn) = self.lstm(x, (h0, c0))

        last_output = output[:, -1, :]

        out = self.act_func(self.fc(last_output))

        return out
    
    def device(self):
        """ Return the device the model is currently on. """
        return next(self.parameters()).device

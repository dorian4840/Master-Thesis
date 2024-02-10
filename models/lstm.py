import torch
from torch import nn


class LSTM(nn.Module):
    """
    Use the pytorch_tcn TCN and add a linear layer at the end to reduce the
    output to a desired length.
    """

    def __init__(self, name, num_input, hidden_size, num_layers, num_output,
                 dropout=0, final_activation='softmax'):

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
        self.fc1 = nn.Linear(hidden_size, num_output[0])
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('sigmoid'))
        self.fc1.bias.data.fill_(0.01)

        self.fc2 = nn.Linear(hidden_size, num_output[1])
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('sigmoid'))
        self.fc2.bias.data.fill_(0.01)

        # self.norm1 = nn.LayerNorm(num_output[0])
        # self.norm2 = nn.LayerNorm(num_output[1])

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

        out_avpu = self.act_func(self.fc1(last_output))
        out_crt = self.act_func(self.fc2(last_output))

        # out_avpu = self.act_func(self.norm1(self.fc1(last_output)))
        # out_crt = self.act_func(self.norm2(self.fc2(last_output)))

        return out_avpu, out_crt
    
    def device(self):
        """ Return the device the model is currently on. """
        return next(self.parameters()).device

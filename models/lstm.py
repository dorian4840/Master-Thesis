import torch
from torch import nn


class LSTM(nn.Module):
    """
    Use the pytorch_tcn TCN and add a linear layer at the end to reduce the
    output to a desired length.
    """

    def __init__(self, num_input, hidden_size, num_layers, num_output,
                 dropout=0):

        super().__init__()

        self.num_input = num_input
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_output = num_output
        self.dropout = dropout

        # LSTM
        self.lstm = nn.LSTM(num_input, hidden_size, num_layers,
                             dropout=dropout, batch_first=True)
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

        # Fully Connected Layer
        self.fc_avpu = nn.Linear(hidden_size, num_output[0])
        nn.init.xavier_normal_(self.fc_avpu.weight)
        self.fc_avpu.bias.data.fill_(0.01)

        self.fc_crt = nn.Linear(hidden_size, num_output[1])
        nn.init.xavier_normal_(self.fc_crt.weight)
        self.fc_crt.bias.data.fill_(0.01)

        # self.norm = nn.LayerNorm(hidden_size)
        self.norm0 = nn.LayerNorm(num_output[0])
        self.norm1 = nn.LayerNorm(num_output[1])
        

        self.act_func = nn.Softmax(dim=-1)


    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(self.device())
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(self.device())

        self.lstm.flatten_parameters()
        output, _ = self.lstm(x, (h0, c0))

        last_output = output[:, -1, :]

        # last_output = self.norm(last_output)

        out_avpu = self.act_func(self.norm0(self.fc_avpu(last_output)))
        out_crt = self.act_func(self.norm1(self.fc_crt(last_output)))

        return out_avpu, out_crt


    def device(self):
        """ Return the device the model is currently on. """
        return next(self.parameters()).device

import torch
from torch import nn
import pytorch_tcn


class TCN(nn.Module):
    """
    Use the pytorch_tcn TCN and add a linear layer at the end to reduce the
    output to a desired length.
    """

    def __init__(self, num_input, num_timesteps, num_channels, kernel_size,
                 dilations=None, dilation_reset=None, dropout=0, causal=False,
                 use_norm='weight_norm', activation='relu', kernel_initializer='xavier_uniform',
                 use_skip_connections=False, input_shape='NLC', num_output=[4, 7]):

        super().__init__()

        self.layers = []

        # TCN model
        temporal_model = pytorch_tcn.TCN(num_inputs=num_input,
                                         num_channels=num_channels,
                                         kernel_size=kernel_size,
                                         dilations=dilations,
                                         dilation_reset=dilation_reset,
                                         dropout=dropout,
                                         causal=causal,
                                         use_norm=use_norm,
                                         activation=activation,
                                         kernel_initializer=kernel_initializer,
                                         use_skip_connections=use_skip_connections,
                                         input_shape=input_shape)

        self.layers.append(temporal_model)

        # Output layers
        self.layers.append(nn.Flatten())
        output_dims = num_channels[-1] * num_timesteps

        # Output layer for the AVPU
        self.linearavpu = nn.Linear(output_dims, num_output[0])
        nn.init.xavier_uniform_(self.linearavpu.weight, gain=nn.init.calculate_gain('sigmoid'))
        self.linearavpu.bias.data.fill_(0.01) # NOTE: Gewoon gevonden online!!!

        # Output layer for the CRT
        self.linearcrt = nn.Linear(output_dims, num_output[1])
        nn.init.xavier_uniform_(self.linearcrt.weight, gain=nn.init.calculate_gain('sigmoid'))
        self.linearcrt.bias.data.fill_(0.01) # NOTE: Gewoon gevonden online!!!

        self.act_func = nn.Softmax(dim=-1)
        self.layers = nn.Sequential(*self.layers)


    def forward(self, x):
        """
        Forward data through the model.
        TODO: Make it ignore missing data.
        """

        out = self.layers(x)
        out_avpu = self.act_func(self.linearavpu(out))
        out_crt = self.act_func(self.linearcrt(out))

        return out_avpu, out_crt


    def device(self):
        """ Return the device the model is currently on. """
        return next(self.parameters()).device

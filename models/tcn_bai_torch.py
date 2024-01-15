import torch
from torch import nn
import pytorch_tcn


class TCN(nn.Module):
    """
    Use the pytorch_tcn TCN and add a linear layer at the end to reduce the
    output to a desired length.
    """

    def __init__(self,
                 num_input,
                 num_timesteps,
                 num_channels,
                 kernel_size,
                 dilations=None,
                 dilation_reset=None,
                 dropout=0,
                 causal=False,
                 use_norm='weight_norm',
                 activation='relu',
                 kernel_initializer='xavier_uniform',
                 use_skip_connections=False,
                 input_shape='NCL',
                 hidden_dims=[500],
                 num_output=2):

        super(TCN, self).__init__()

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

        for hidden in hidden_dims:
            linear = nn.Linear(output_dims, hidden)
            nn.init.xavier_uniform_(linear.weight, gain=nn.init.calculate_gain('relu'))
            linear.bias.data.fill_(0.01) # NOTE: Gewoon gevonden online!!!

            self.layers.append(linear)
            self.layers.append(nn.ReLU())
            output_dims = hidden

        linear = nn.Linear(output_dims, num_output)
        nn.init.xavier_uniform_(linear.weight, gain=nn.init.calculate_gain('sigmoid'))
        linear.bias.data.fill_(0.01) # NOTE: Gewoon gevonden online!!!

        self.layers.append(linear)
        self.layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*self.layers)


    def forward(self, x):
        """
        Forward data through the model.
        TODO: Make it ignore missing data.
        """

        out = self.layers(x)
        return out



# data = torch.rand((16, 454, 4))

# model = TCN(
#     num_input=454,
#     num_timesteps=4,
#     num_channels=[250, 100, 50],
#     kernel_size=2,
#     dilations=[1, 2, 4],
#     dilation_reset=None,
#     dropout=0.2,
#     causal=True,
#     use_norm='weight_norm',
#     activation='relu',
#     kernel_initializer='xavier_uniform',
#     use_skip_connections=False,
#     input_shape='NCL',
#     hidden_dims=[100, 25],
#     num_output=2
# )

# print(model)

# print(data.shape)
# out = model(data)
# print(out[0])

# model = pytcn.TCN(num_inputs=10,        # Equal to num input features
#                   num_channels=[15, 15, 15],    # Must match number of dilations
#                   kernel_size=2,
#                   dilations=[1,2,3],    # If None, 2^(1...n) for residual blocks.
#                   dilation_reset=3,     # dilation 3 is max for input length 4
#                   dropout=0.2,
#                   causal=False,          # Causal is important for real-time predictions
#                   use_norm='weight_norm',
#                   activation='relu',
#                   kernel_initializer='xavier_uniform',
#                   use_skip_connections=False,   # from WaveNet architecture
#                   input_shape='NCL')    # Batch size, features, time


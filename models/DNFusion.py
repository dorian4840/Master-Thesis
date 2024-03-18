"""
This file contains the main fusion model of the thesis.
"""

import torch
from torch import nn
import pytorch_tcn


class DNFusion(nn.Module):
    """
    Fusion model that combines a Temporal Convolutional Network (TCN) and a
    Long Short-Term Memory model (LSTM) using a Multi-Modal Transfer Module (MMTM).
    """

    def __init__(self, model1_config, model2_config, num_outputs=[4, 7],
                 mmtm_ratio=1, mask_mean=0, mask_std=1, threshold=0.1,
                 fusion=True, warmup=5):

        super().__init__()

        assert len(model1_config['num_channels']) == len(model1_config['num_channels']),\
            'Number of TCN channels must be equal to each other'

        self.num_inputs1 = model1_config['num_inputs']
        self.num_timesteps1 = model1_config['num_timesteps']
        self.num_channels1 = model1_config['num_channels']
        self.num_inputs2 = model2_config['num_inputs']
        self.num_timesteps2 = model2_config['num_timesteps']
        self.num_channels2 = model2_config['num_channels']
        self.num_layers = len(self.num_channels1)
        self.fusion = fusion
        self.warmup = warmup

        # Init Gaussian
        self.mask = Mask(mask_mean, mask_std, threshold)

        # Init MMTM
        if self.fusion:
            self.mmtm = self.init_mmtm(model1_config['num_channels'],
                                       model2_config['num_channels'],
                                       ratio=mmtm_ratio)

        # Init models
        self.model1 = self.init_tcn(model1_config)
        self.model2 = self.init_tcn(model2_config)
        self.flatten = nn.Flatten()

        # Final linear layers
        model1_output_dims = self.num_channels1[-1] * self.num_timesteps1
        model2_output_dims = self.num_channels2[-1] * self.num_timesteps2

        self.fc_avpu = nn.Linear(model1_output_dims + model2_output_dims, num_outputs[0])
        nn.init.xavier_uniform_(self.fc_avpu.weight, gain=nn.init.calculate_gain('sigmoid'))
        self.fc_avpu.bias.data.fill_(0.01)
        self.norm0 = nn.LayerNorm(num_outputs[0])

        self.fc_crt = nn.Linear(model1_output_dims + model2_output_dims, num_outputs[1])
        nn.init.xavier_uniform_(self.fc_crt.weight, gain=nn.init.calculate_gain('sigmoid'))
        self.fc_crt.bias.data.fill_(0.01)
        self.norm1 = nn.LayerNorm(num_outputs[1])

        self.act_fn = nn.Softmax(dim=-1)


    def init_mmtm(self, num_channels1, num_channels2, ratio=1):
        """
        Initialize the MultiModal Transfer Modules.
        """

        modules = nn.ModuleList([MMTM(num_channels1[i], num_channels2[i], ratio) for \
                                 i in range(self.num_layers)])

        return modules


    def init_tcn(self, config):
        """
        Initialize the Temporal Convolutional Network
        """

        model = pytorch_tcn.TCN(num_inputs=config['num_inputs'],
                                num_channels=config['num_channels'],
                                kernel_size=config['kernel_size'],
                                dilations=config['dilations'],
                                dilation_reset=config['dilation_reset'],
                                dropout=config['dropout'], 
                                causal=config['causal'],
                                use_norm=config['use_norm'],
                                activation=config['activation'],
                                kernel_initializer=config['kernel_initializer'],
                                use_skip_connections=False,
                                input_shape='NLC')

        return model.get_submodule('network')


    def forward(self, clinical_input, vital_input):
        """
        Forward the two datastreams through the corresponding model and exchange
        data using an MMTM. After the model, linear layers are used to create
        the appropriate output shapes.
        NOTE:  MMTM requires bs x features x length
        """

        # For-loop through the models' layers.
        for i in range(self.num_layers):

            # Forward model 1
            clinical_input = clinical_input.transpose(1, 2)
            clinical_input, _ = self.model1[i](clinical_input)
            clinical_input = clinical_input.transpose(1, 2)

            # Forward model 2
            vital_input = vital_input.transpose(1, 2)
            vital_input, _ = self.model2[i](vital_input)
            vital_input = vital_input.transpose(1, 2)

            # Fusion with MMTM
            if self.fusion:

                clinical_input = clinical_input.transpose(1, 2)
                vital_input = vital_input.transpose(1, 2)

                if self.warmup:
                    model1_out, model2_out = self.mmtm[i](clinical_input, vital_input)
                    clinical_input = clinical_input * model1_out
                    vital_input = vital_input * model2_out

                elif self.mask(i) > 0:
                    model1_out, model2_out = self.mmtm[i](clinical_input, vital_input)
                    clinical_input = clinical_input * model1_out * self.mask(i)
                    vital_input = vital_input * model2_out * self.mask(i)

                clinical_input = clinical_input.transpose(1, 2)
                vital_input = vital_input.transpose(1, 2)


        model1_output = self.flatten(clinical_input) # flatten output
        model2_output = self.flatten(vital_input) # flatten output

        # Concat outputs and divide into APVU and CRT part
        concatenated_output = torch.concat((model1_output, model2_output), dim=1)
        avpu_output = self.act_fn(self.norm0(self.fc_avpu(concatenated_output)))
        crt_output = self.act_fn(self.norm1(self.fc_crt(concatenated_output)))

        return avpu_output, crt_output


    def start_warmup(self):
        """
        Start MMTM warmup and freeze every layer except for the MMTM.
        """
        print('Warming up fusion module...')
        for name, param in self.named_parameters():
            if 'mmtm' not in name:
                param.requires_grad = False


    def end_warmup(self):
        """
        End MMTM warmup and unfreeze all layers.
        """
        print('Warm up finished...')
        self.warmup = False
        for name, param in self.named_parameters():
            # if 'std' not in name:
            param.requires_grad = True


    def device(self):
        """ Return the device the model is currently on. """
        return next(self.parameters()).device


class Mask(nn.Module):
    def __init__(self, mean, std, threshold=0.1):

        super().__init__()
        self.mean = nn.Parameter(torch.tensor(float(mean)), requires_grad=True)
        self.std = nn.Parameter(torch.tensor(float(std)), requires_grad=True)
        # self.std = nn.Parameter(torch.tensor(float(std)), requires_grad=False)
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=False)


    def get_probs(self, values):
        """ Calculate the Gaussian probability at certain value on the x-axis. """
        probs = torch.tensor([-0.5 * ((v - self.mean)**2 / self.std**2) for v in values])
        return torch.exp(probs)
    

    def prob(self, value, amplifier=100):
        p = torch.exp(-0.5 * ((value - (self.mean*amplifier))**2 / (self.std*amplifier)**2))
        
        # if value <= self.mean*amplifier and p >= self.threshold:
        #     return p
        # elif value > self.mean*amplifier and p >= self.threshold:
        #     return p
        # else:
        #     return torch.tensor(0.0)

        if p >= self.threshold:
            return p
        return torch.tensor(0.0)


    def forward(self, value):
        """
        Calculate the probability and mask for each index on the fusion depth axis.
        """
        return self.prob(value)


class MMTM(nn.Module):
    def __init__(self, dim_visual, dim_skeleton, ratio):

        super(MMTM, self).__init__()

        dim = dim_visual + dim_skeleton
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, visual, skeleton):
        """ Forward both data streams through the MMTM. """

        squeeze_array = []

        for tensor in [visual, skeleton]:
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_array.append(torch.mean(tview, dim=-1))

        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        dim_diff = len(visual.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

        return vis_out, sk_out

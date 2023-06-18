from torch import optim
from lop.algos.gnt_mean_update import GnT 
import torch.nn.functional as F

import numpy as np
import torch

class MeanUpdate(object):
    def __init__(
            self,
            net,
            step_size=0.001,
            loss='mse',
            opt='sgd',
            beta=0.9,
            beta_2=0.999,
            replacement_rate=0.001,
            decay_rate=0.9,
            device='cpu',
            maturity_threshold=100,
            util_type='contribution',
            init='kaiming',
            accumulate=False,
            momentum=0,
            outgoing_random=False,
            weight_decay=0,
    ):
        self.net = net

        # define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, momentum=momentum, weight_decay=weight_decay)
        # elif opt == 'adam':
        #     self.opt = AdamGnT(self.net.parameters(), lr=step_size, betas=(beta, beta_2), weight_decay=weight_decay)

        # define the loss function
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[loss]

        # a placeholder
        self.previous_features = None

        # define the generate-and-test object for the given network
        self.gnt = None
        self.gnt = GnT(
            net=self.net.layers,
            hidden_activation=self.net.act_type,
            opt=self.opt,
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            maturity_threshold=maturity_threshold,
            util_type=util_type,
            device=device,
            loss_func=self.loss_func,
            init=init,
            accumulate=accumulate,
        )

    def learn(self, x, target):
        """
        Learn using one step of gradient-descent and generate-&-test
        :param x: input
        :param target: desired output
        :return: loss
        """
        # do a forward pass and get the hidden activations
        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target)
        self.previous_features = features

        # store previous version of parameters before gradient step
        prev_named_params = [(name, param.detach().clone()) for name, param in 
                             self.net.named_parameters()]

        # do the backward pass and take a gradient step
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.gnt.util_type == 'mean_update':
            named_params = [(name, param.detach().clone()) for name, param in 
                            self.net.named_parameters()]
            col_norms = [torch.norm(named_params[i][1] - prev_named_params[i][1], p=2, dim=0)
                         for i in range(2, len(named_params), 2)]

        # take a generate-and-test step
        self.opt.zero_grad()
        if type(self.gnt) is GnT:
            self.gnt.gen_and_test(self.previous_features, col_norms)

        if self.loss_func == F.cross_entropy:
            return loss.detach(), output.detach()

        return loss.detach()
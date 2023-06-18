from torch import optim
from lop.algos.gnt_natimp import GnT 
import torch.nn.functional as F

import numpy as np
import torch

class NatImp(object):
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
            feature_ablation_rate=1
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

        self.feature_ablation_rate = feature_ablation_rate

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

        # do the backward pass and take a gradient step
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # First step in test mechanism is to do neuron ablation
        if self.gnt.util_type == 'natural_importance':
            named_params = [(name, param) for name, param in self.net.named_parameters()]
            all_ablation_losses = []
            for i in range(2, len(named_params), 2):
                layer_ablation_losses = []
                for j in range(named_params[i][1].shape[1]):
                    output_, features_ = self.net.predict(x=x, weight_to_zero=i, col_to_zero=j)
                    loss_ = self.loss_func(output_, target)
                    layer_ablation_losses.append(loss_.item())
                # Sometimes some of the layer_ablation_losses are smaller than 
                # the full loss, so finding the squared error between them as in
                # https://arxiv.org/pdf/1906.10771.pdf doesn't make sense (
                # squaring removes the sign).
                loss_rep = loss * torch.ones(named_params[i][1].shape[1])
                all_ablation_losses.append(
                    loss_rep - torch.tensor(layer_ablation_losses))
        
        elif self.gnt.util_type == 'sparse_natural_importance':
            named_params = [(name, param) for name, param in self.net.named_parameters()]
            all_ablation_losses = []
            for i in range(2, len(named_params), 2):
                layer_ablation_losses = []
                for j in range(named_params[i][1].shape[1]):
                    if torch.rand(1) <= self.feature_ablation_rate:
                        output_, features_ = self.net.predict(x=x, weight_to_zero=i, col_to_zero=j)
                        loss_ = self.loss_func(output_, target)
                        layer_ablation_losses.append(loss_.item())
                    else:
                        layer_ablation_losses.append(loss)
                # If a feature is not ablated, then the loss is the same as the
                # full loss. This means the average util for a feature will get 
                # pulled closer to zero, but this is fine since the same happens 
                # for all features and we compare features against one another,
                # not a threshold, when deciding which to reset.
                loss_rep = loss * torch.ones(named_params[i][1].shape[1])
                all_ablation_losses.append(
                    loss_rep - torch.tensor(layer_ablation_losses))

        # take a generate-and-test step
        self.opt.zero_grad()
        if type(self.gnt) is GnT:
            self.gnt.gen_and_test(self.previous_features, all_ablation_losses)

        if self.loss_func == F.cross_entropy:
            return loss.detach(), output.detach()

        return loss.detach()


class Taylor(object):
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
            feature_ablation_rate=1
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

        self.feature_ablation_rate = feature_ablation_rate

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

        # do the backward pass and take a gradient step
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()        

        # First step in test mechanism is to calculate approximate gradients 
        # for when each feature is ablated.
        if self.gnt.util_type == 'natimp_taylor':
            named_params = [(name, param) for name, param in self.net.named_parameters()]
            named_grads = [(name, param.grad) for name, param in self.net.named_parameters()]
            all_ablation_losses = []
            for i in range(2, len(named_grads), 2):
                layer_ablation_losses = []
                for j in range(named_grads[i][1].shape[1]):
                    W = named_params[i][1].detach().clone()
                    W_tilde = named_params[i][1].detach().clone()
                    W_tilde[:, j] = 0
                    loss_ = loss + named_grads[i][1].detach().clone() @ (W_tilde - W).T
                    layer_ablation_losses.append(loss_.item())
                loss_rep = loss * torch.ones(named_params[i][1].shape[1])
                all_ablation_losses.append(
                    loss_rep - torch.tensor(layer_ablation_losses))

        # take a generate-and-test step
        self.opt.zero_grad()
        if type(self.gnt) is GnT:
            self.gnt.gen_and_test(self.previous_features, all_ablation_losses)

        if self.loss_func == F.cross_entropy:
            return loss.detach(), output.detach()

        return loss.detach()

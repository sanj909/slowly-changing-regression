import torch
import torch.nn as nn


class FFNN(nn.Module):
    """
    A feed forward neural network with just one hidden layer.
    This network is used as the learning network in the Slowly Changing Regression problem
    """
    def __init__(self, input_size, num_features=5, num_outputs=1, hidden_activation='relu'):
        super(FFNN, self).__init__()
        self.num_inputs = input_size
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.act_type = hidden_activation

        # define the hidden activation
        self.hidden_activation = {'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU, 'selu': nn.SELU,
                                  'swish': nn.SiLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}[self.act_type]

        # define the architecture
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, num_features))
        self.layers.append(self.hidden_activation())
        self.layers.append(nn.Linear(num_features, num_outputs))

        # initialize the input weights
        self.layers[0].bias.data.fill_(0.0)
        if hidden_activation in ['sigmoid', 'relu', 'tanh', 'leaky_relu']:
            nn.init.kaiming_uniform_(self.layers[0].weight, nonlinearity=hidden_activation)
        elif hidden_activation in ['swish', 'elu']:
            nn.init.kaiming_uniform_(self.layers[0].weight, nonlinearity='relu')
        # initialize the output weights
        nn.init.kaiming_uniform_(self.layers[-1].weight, nonlinearity='linear')
        self.layers[-1].bias.data.fill_(0.0)

    def predict(self, x):
        """
        Forward pass
        :param x: input
        :return: estimated output
        """
        features = self.layers[1](self.layers[0](x))
        out = self.layers[-1](features)
        return out, [features]
    

# ------------------------------------------------------------------------------
class FFNN_abl(nn.Module):
    """
    A feed forward neural network with just one hidden layer.
    This network is used as the learning network in the Slowly Changing Regression problem
    """
    def __init__(self, input_size, num_features=5, num_outputs=1, hidden_activation='relu'):
        super(FFNN_abl, self).__init__()
        self.num_inputs = input_size
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.act_type = hidden_activation

        # define the hidden activation
        self.hidden_activation = {'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU, 'selu': nn.SELU,
                                  'swish': nn.SiLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}[self.act_type]

        # define the architecture
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, num_features))
        self.layers.append(self.hidden_activation())
        self.layers.append(nn.Linear(num_features, num_outputs))

        # initialize the input weights
        self.layers[0].bias.data.fill_(0.0)
        if hidden_activation in ['sigmoid', 'relu', 'tanh', 'leaky_relu']:
            nn.init.kaiming_uniform_(self.layers[0].weight, nonlinearity=hidden_activation)
        elif hidden_activation in ['swish', 'elu']:
            nn.init.kaiming_uniform_(self.layers[0].weight, nonlinearity='relu')
        # initialize the output weights
        nn.init.kaiming_uniform_(self.layers[-1].weight, nonlinearity='linear')
        self.layers[-1].bias.data.fill_(0.0)

    def predict(self, x, weight_to_zero=None, col_to_zero=None):
        """
        Forward pass
        :param x: input
        :return: estimated output
        """
        if weight_to_zero is None:
            features = self.layers[1](self.layers[0](x))
            out = self.layers[-1](features)
            return out, [features]
        else:
            old_weight = torch.clone(self.layers[weight_to_zero].weight)
            
            mask = torch.ones(old_weight.shape[1])
            mask[col_to_zero] = 0
            mask = mask.clone().detach().reshape(1, -1)

            modded_weight = old_weight.clone()
            modded_weight[mask == 0] = 0

            self.layers[weight_to_zero].weight.data.copy_(modded_weight)

            # Perform prediction with modified weights
            features = self.layers[1](self.layers[0](x))
            out = self.layers[-1](features)

            # Restore original weights
            self.layers[weight_to_zero].weight.data.copy_(old_weight)

            return out, [features]
# ------------------------------------------------------------------------------

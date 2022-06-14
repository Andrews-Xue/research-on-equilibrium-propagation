
import torch

from lib import cost


def create_activations(name, n_layers):

    if name == 'relu':
        phi_l = torch.relu
    elif name == "leaky_relu":
        phi_l = torch.nn.functional.leaky_relu
    elif name == 'softplus':
        phi_l = torch.nn.functional.softplus
    elif name == 'tanh':
        phi_l = torch.tanh
    elif name == 'sigmoid':
        phi_l = torch.sigmoid
    elif name == 'hard_sigmoid':
        def phi_l(x): torch.clamp(x, min=0, max=1)
    else:
        raise ValueError(f'Nonlinearity \"{name}\" not defined.')

    return [lambda x: x] + [phi_l] * (n_layers - 1)


def create_cost(name, beta):

    if name == "squared_error":
        return cost.SquaredError(beta)
    elif name == "cross_entropy":
        return cost.CrossEntropy(beta)
    else:
        raise ValueError("Cost function \"{}\" not defined".format(name))


def create_optimizer(model, name, **kwargs):

    if name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), **kwargs)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), **kwargs)
    else:
        raise ValueError("Optimizer \"{}\" undefined".format(name))

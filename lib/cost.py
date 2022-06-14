import abc

import torch


class CEnergy(abc.ABC):

    def __init__(self, beta):
        super(CEnergy, self).__init__()
        self.beta = beta
        self.target = None

    @abc.abstractmethod
    def compute_energy(self, u_last):
 
        return

    def set_target(self, target):

        self.target = target


class CrossEntropy(CEnergy):

    def __init__(self, beta):
        super(CrossEntropy, self).__init__(beta)

    def compute_energy(self, u_last):

        loss = torch.nn.functional.cross_entropy(u_last, self.target, reduction='none')
        return self.beta * loss

    def set_target(self, target):
        if target is None:
            self.target = None
        else:
            # Need to transform target for the F.cross_entropy function
            self.target = target.argmax(dim=1)


class SquaredError(CEnergy):

    def __init__(self, beta):
        super(SquaredError, self).__init__(beta)

    def compute_energy(self, u_last):

        loss = torch.nn.functional.mse_loss(u_last, self.target.float(), reduction='none')
        return self.beta * 0.5 * torch.sum(loss, dim=1)

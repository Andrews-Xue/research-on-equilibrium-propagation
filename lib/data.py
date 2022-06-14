import torch
import torchvision
from torchvision import datasets, transforms

def _one_hot_ten(label):

    return torch.nn.functional.one_hot(torch.tensor(label), num_classes=10)


def create_mnist_loaders(batch_size):


    # Load train and test MNIST datasets
    mnist_train = datasets.MNIST('../data/', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,)),
                                 ]),
                                 target_transform=_one_hot_ten
                                 )

    mnist_test = datasets.MNIST('../data/', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                ]),
                                target_transform=_one_hot_ten
                                )

    
    # For GPU acceleration store dataloader in pinned (page-locked) memory
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # Create the dataloader objects
    train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, drop_last=True, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, drop_last=True, shuffle=False, **kwargs)

    return train_loader, test_loader

def create_Fmnist_loaders(batch_size):
    
    # Load train and test FMNIST datasets

    Fmnist_train = torchvision.datasets.FashionMNIST(root='./data/Fashion-MNIST', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,)),
                                 ]),
                                 target_transform=_one_hot_ten)

    Fmnist_test = torchvision.datasets.FashionMNIST('./data/Fashion-MNIST', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                ]),
                                target_transform=_one_hot_ten
                                )

    
    # For GPU acceleration store dataloader in pinned (page-locked) memory
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # Create the dataloader objects
    Ftrain_loader = torch.utils.data.DataLoader(
        Fmnist_train, batch_size=batch_size, drop_last=True, shuffle=True, **kwargs)

    Ftest_loader = torch.utils.data.DataLoader(
        Fmnist_test, batch_size=batch_size, drop_last=True, shuffle=False, **kwargs)

    return Ftrain_loader, Ftest_loader

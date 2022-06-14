import argparse
import json
import logging
import sys

import torch

from lib import config, data, energy, train, utils


def load_default_config(energy):

    if energy == "restr_hopfield":
        default_config = "etc/energy_restr_hopfield.json"
    elif energy == "cond_gaussian":
        default_config = "etc/energy_cond_gaussian.json"
    else:
        raise ValueError("Energy based model \"{}\" not defined.".format(energy))

    with open(default_config) as config_json_file:
        cfg = json.load(config_json_file)

    return cfg


def parse_shell_args(args):

    parser = argparse.ArgumentParser(
        description="Train an energy-based model on MNIST using Equilibrium Propagation."
    )

    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS,
                        help="Size of mini batches during training.")
    parser.add_argument("--c_energy", choices=["cross_entropy", "squared_error"],
                        default=argparse.SUPPRESS, help="Supervised learning cost function.")
    parser.add_argument("--dimensions", type=int, nargs="+",
                        default=argparse.SUPPRESS, help="Dimensions of the neural network.")
    parser.add_argument("--energy", choices=["cond_gaussian", "restr_hopfield"],
                        default="cond_gaussian", help="Type of energy-based model.")
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS,
                        help="Number of epochs to train.")
    parser.add_argument("--fast_ff_init", action='store_true', default=argparse.SUPPRESS,
                        help="Flag to enable fast feedforward initialization.")
    parser.add_argument("--learning_rate", type=float, default=argparse.SUPPRESS,
                        help="Learning rate of the optimizer.")
    parser.add_argument("--log_dir", type=str, default="",
                        help="Subdirectory within ./log/ to store logs.")
    parser.add_argument("--nonlinearity", choices=["leaky_relu", "relu", "sigmoid", "tanh", "softplus"],
                        default=argparse.SUPPRESS, help="Nonlinearity between network layers.")
    parser.add_argument("--optimizer", choices=["adam", "adagrad", "sgd"],
                        default=argparse.SUPPRESS, help="Optimizer used to train the model.")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS,
                        help="Random seed for pytorch")

    return vars(parser.parse_args(args))


def run_energy_model_mnist(cfg):

    # Initialize seed if specified (might slow down the model)
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])

    # Create the cost function to be optimized by the model
    c_energy = utils.create_cost(cfg['c_energy'], cfg['beta'])

    # Create activation functions for every layer as a list
    phi = utils.create_activations(cfg['nonlinearity'], len(cfg['dimensions']))

    # Initialize energy based model
    if cfg["energy"] == "restr_hopfield":
        model = energy.RestrictedHopfield(
            cfg['dimensions'], c_energy, cfg['batch_size'], phi).to(config.device)
    elif cfg["energy"] == "cond_gaussian":
        model = energy.ConditionalGaussian(
            cfg['dimensions'], c_energy, cfg['batch_size'], phi).to(config.device)
    else:
        raise ValueError(f'Energy based model \"{cfg["energy"]}\" not defined.')

    # Define optimizer (may include l2 regularization via weight_decay)
    w_optimizer = utils.create_optimizer(model, cfg['optimizer'],  lr=cfg['learning_rate'])

    # Create torch data loaders with the MNIST data set

    Fmnist_train, Fmnist_test = data.create_Fmnist_loaders(cfg['batch_size'])
 
    logging.info("Start training with parametrization:\n{}".format(
        json.dumps(cfg, indent=4, sort_keys=True)))
    '''
    for epoch in range(1, cfg['epochs'] + 1):
        # Training
        train.train(model, mnist_train, cfg['dynamics'], w_optimizer, cfg["fast_ff_init"])

        # Testing
        test_acc, test_energy = train.test(model, mnist_test, cfg['dynamics'], cfg["fast_ff_init"])

        # Logging
        logging.info(
            "epoch: {} \t test_acc: {:.4f} \t mean_E: {:.4f}".format(
                epoch, test_acc, test_energy)
        )
    '''
    for epoch in range(1, cfg['epochs'] + 1):
        # Training
        train.train(model, Fmnist_train, cfg['dynamics'], w_optimizer, cfg["fast_ff_init"])

        # Testing
        test_acc, test_energy = train.test(model, Fmnist_test, cfg['dynamics'], cfg["fast_ff_init"])

        # Logging
        logging.info(
            "epoch: {} \t test_acc: {:.4f} \t mean_E: {:.4f}".format(
                epoch, test_acc, test_energy)
        )
    
if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(1)
    # Parse shell arguments as input configuration
    user_config = parse_shell_args(sys.argv[1:])

    # Load default parameter configuration from file for the specified energy-based model
    cfg = load_default_config(user_config["energy"])

    # Overwrite default parameters with user configuration where applicable
    cfg.update(user_config)

    # Setup global logger and logging directory
    config.setup_logging(cfg["energy"] + "_" + cfg["c_energy"] + "_" + cfg["dataset"],
                         dir=cfg['log_dir'])

    # Run the script using the created paramter configuration
    run_energy_model_mnist(cfg)

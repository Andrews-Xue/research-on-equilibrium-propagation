
import logging

import torch

from lib import config


def predict_batch(model, x_batch, dynamics, fast_init):

    # Initialize the neural state variables
    model.reset_state()

    # Clamp the input to the test sample, and remove nudging from ouput
    model.clamp_layer(0, x_batch.view(-1, model.dimensions[0]))
    model.set_C_target(None)

    # Generate the prediction
    if fast_init:
        model.fast_init()
    else:
        model.u_relax(**dynamics)

    return torch.nn.functional.softmax(model.u[-1].detach(), dim=1)


def test(model, test_loader, dynamics, fast_init):

    test_E, correct, total = 0.0, 0.0, 0.0

    for x_batch, y_batch in test_loader:
        # Prepare the new batch
        x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)

        # Extract prediction as the output unit with the strongest activity
        output = predict_batch(model, x_batch, dynamics, fast_init)
        prediction = torch.argmax(output, 1)

        with torch.no_grad():
            # Compute test batch accuracy, energy and store number of seen batches
            correct += float(torch.sum(prediction == y_batch.argmax(dim=1)))
            test_E += float(torch.sum(model.E))
            total += x_batch.size(0)

    return correct / total, test_E / total


def train(model, train_loader, dynamics, w_optimizer, fast_init):

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)

        # Reinitialize the neural state variables
        model.reset_state()

        # Clamp the input to the training sample
        model.clamp_layer(0, x_batch.view(-1, model.dimensions[0]))

        # Free phase
        if fast_init:
            # Skip the free phase using fast feed-forward initialization instead
            model.fast_init()
            free_grads = [torch.zeros_like(p) for p in model.parameters()]
        else:
            # Run free phase until settled to fixed point and collect the free phase derivates
            model.set_C_target(None)
            dE = model.u_relax(**dynamics)
            free_grads = model.w_get_gradients()

        # Run nudged phase until settled to fixed point and collect the nudged phase derivates
        model.set_C_target(y_batch)
        dE = model.u_relax(**dynamics)
        nudged_grads = model.w_get_gradients()

        # Optimize the parameters using the contrastive Hebbian style update
        model.w_optimize(free_grads, nudged_grads, w_optimizer)

        # Logging key statistics
        if batch_idx % (len(train_loader) // 10) == 0:

            # Extract prediction as the output unit with the strongest activity
            output = predict_batch(model, x_batch, dynamics, fast_init)
            prediction = torch.argmax(output, 1)

            # Log energy and batch accuracy
            batch_acc = float(torch.sum(prediction == y_batch.argmax(dim=1))) / x_batch.size(0)
            logging.info('{:.0f}%:\tE: {:.2f}\tdE {:.2f}\tbatch_acc {:.4f}'.format(
                100. * batch_idx / len(train_loader), torch.mean(model.E), dE, batch_acc))

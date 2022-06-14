# research on equilibrium propagation

“RESEARCH ON EQUILIBRIUM PROPAGATION ALGORITHMS AND A CASE STUDY”

 The equilibrium propagation algorithm is a machine learning framework based on an energy model. The algorithm makes up for the shortcomings of the traditional backpropagation algorithm and the contrastive Hebbian learning algorithm in the inconsistent computation used in backward and forward propagation. This makes the gradient propagation algorithm more reasonable in terms of biological interpretability. On the basis of referring to many papers, this paper makes a comprehensive investigation of the research background and state-of-the-art equilibrium propagation algorithm and finds some problems with equilibrium propagation. The article introduces the energy-based machine learning model and the gradient algorithm in detail, and makes a cross-sectional comparison between the equilibrium propagation algorithm and similar other algorithms. And it also points out the differences between the algorithms, and demonstrates the advantages and disadvantages of the equilibrium propagation algorithm. Based on the research, a case implementation of the equilibrium propagation algorithm is presented, and the effect of different data sets, loss functions, energy functions and activation functions is tested in the experiments. 

You can run the models using the `run_energy_model_mnist.py` script which provides the following options:

```
python run_energy_model_mnist.py -h
usage: run_energy_model_mnist.py [-h] [--batch_size BATCH_SIZE]
                                 [--c_energy {cross_entropy,squared_error}]
                                 [--dimensions DIMENSIONS [DIMENSIONS ...]]
                                 [--energy {cond_gaussian,restr_hopfield}]
                                 [--epochs EPOCHS] [--fast_ff_init]
                                 [--learning_rate LEARNING_RATE]
                                 [--log_dir LOG_DIR]
                                 [--nonlinearity {leaky_relu,relu,sigmoid,tanh}]
                                 [--optimizer {adam,adagrad,sgd}]
                                 [--seed SEED]

Train an energy-based model on MNIST using Equilibrium Propagation.

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Size of mini batches during training.
  --c_energy {cross_entropy,squared_error}
                        Supervised learning cost function.
  --dimensions DIMENSIONS [DIMENSIONS ...]
                        Dimensions of the neural network.
  --energy {cond_gaussian,restr_hopfield}
                        Type of energy-based model.
  --epochs EPOCHS       Number of epochs to train.
  --fast_ff_init        Flag to enable fast feedforward initialization.
  --learning_rate LEARNING_RATE
                        Learning rate of the optimizer.
  --log_dir LOG_DIR     Subdirectory within ./log/ to store logs.
  --nonlinearity {leaky_relu,relu,sigmoid,tanh}
                        Nonlinearity between network layers.
  --optimizer {adam,adagrad,sgd}
                        Optimizer used to train the model.
  --seed SEED           Random seed for pytorch
```


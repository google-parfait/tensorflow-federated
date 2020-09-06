# Federated Optimization

This directory contains source code for evaluating federated learning with
different optimizers on various models and tasks. The code was developed for a
paper, "Adaptive Federated Optimization"
([arXiv link](https://arxiv.org/abs/2003.00295)). For a more general look at
using TensorFlow Federated for research, see
[Using TFF for Federated Learning Research](https://www.tensorflow.org/federated/tff_for_research).

Some pip packages are required by this library, and may need to be installed:

```
pip install absl-py
pip install attr
pip install dm-tree
pip install numpy
pip install pandas
pip install tensorflow
pip install tensorflow-federated
```

We also require [Bazel](https://www.bazel.build/) in order to run the code.
Please see the guide
[here](https://docs.bazel.build/versions/master/install.html) for installation
instructions.

### Directory structure

This directory is broken up into six task directories. Each task directory
contains task-specific libraries (such as libraries for loading the correct
dataset), as well as libraries for performing federated and non-federated
(centralized) training. These are in the `optimization/{task}` folders.

A single binary for running these tasks can be found at
`main/federated_trainer.py`. This binary will, according to `absl` flags, run
any of the six task-specific federated training libraries.

There is also a `shared` directory with utilities specific to these experiments,
such as implementations of metrics used for evaluation.

### Example usage

Suppose we wish to train a convolutional network on EMNIST for purposes of
character recognition (`emnist_cr`), using federated optimization. Various
aspects of the federated training procedure can be customized via `absl` flags.
For example, from this directory one could run:

```
bazel run main:federated_trainer -- --task=emnist_cr --total_rounds=100
--client_optimizer=sgd --client_learning_rate=0.1 --client_batch_size=20
--server_optimizer=sgd --server_learning_rate=1.0 --clients_per_round=10
--client_epochs_per_round=1 --experiment_name=emnist_fedavg_experiment
```

This will run 100 communication rounds of federated training, using SGD on both
the client and server, with learning rates of 0.1 and 1.0 respectively. The
experiment uses 10 clients in each round, and performs 1 training epoch on each
client's dataset. Each client will use a batch size of 10 The `experiment_name`
flag is for the purposes of writing metrics.

To try using Adam at the server, we could instead set `--server_optimizer=adam`.
Other parameters that can be set include the batch size on the clients, the
momentum parameters for various optimizers, and the number of total
communication rounds.

### Task and dataset summary

Below we give a summary of the datasets, tasks, and models used in this
directory.

<!-- mdformat off(This table is sensitive to automatic formatting changes) -->

| Directory        | Dataset        | Model                             | Task Summary              |
|------------------|----------------|-----------------------------------|---------------------------|
| cifar100         | [CIFAR-100](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data)      | ResNet-18 (with GroupNorm layers) | Image classification      |
| emnist           | [EMNIST](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data)         | CNN (with dropout)                | Digit recognition         |
| emnist_ae        | [EMNIST](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data)         | Bottleneck network                | Autoencoder               |
| shakespeare      | [Shakespeare](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/shakespeare/load_data)    | RNN with 2 LSTM layers            | Next-character prediction |
| stackoverflow    | [Stack Overflow](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data) | RNN with 1 LSTM layer             | Next-word prediction      |
| stackoverflow_lr | [Stack Overflow](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data) | Logistic regression classifier    | Tag prediction            |

<!-- mdformat on -->

### Using different optimizers

In our work, we compare 5 primary optimization methods: **FedAvg**, **FedAvgM**,
**FedAdagrad**, **FedAdam**, and **FedYogi**. The first two use SGD on both
client and server (with **FedAvgM** using server momentum of 0.9) and the last
three use an adaptive optimizer on the server. To recreate our experimental
results for each optimizer, use the following optimizer-specific flags:

*   **FedAvg**: `--server_optimizer=sgd --server_sgd_momentum=0.0`
*   **FedAvgM**: `--server_optimizer=sgd --server_sgd_momentum=0.9`
*   **FedAdagrad**: `--server_optimizer=adagrad
    --server_adagrad_initial_accumulator_value=0.0`
*   **FedAdam**: `--server_optimizer=adam`
*   **FedYogi**: `--server_optimizer=yogi
    --server_yogi_initial_accumulator_value=0.0`

Note that for adaptive optimizers, one should also set the parameter tau in the
full description of our algorithms (see the accompanying paper). This parameter
is referred to as epsilon in our code, and can be set via `--server_{adaptive
optimizer}_epsilon={tau value}`. In general, we recommend a value of at least
0.001 in most tasks. The best values for each task/optimizer are fully
documented in the accompanying paper.

For FedAdagrad and FedYogi, we use implementations of Adagrad and Yogi that
allow one to select the `initial_accumulator_value` (see the Keras documentation
on
[Adagrad](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adagrad)).
For all experiments, we used initial accumulator values of 0 (which is the
implicit value set by default in the Keras implementation of
[Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)).
While this can be tuned, we recommend first tuning other values, especially the
`epsilon` value.

In all of the cases above, the `--client_learning_rate` and
`--server_learning_rate` must be set as well. For a detailed reference of the
best hyperparameters for each optimizer and task, see the appendix in our
accompanying paper.

### Other hyperparameters and reproducibility

All other hyperparameters are set by default to the values used in the `Constant
learning rates` experiments of Section 5 in our paper. This includes the batch
size, the number of clients per round, the number of client epochs, and model
parameter flags. While they can be set for different behavior (such as varying
the number of client epochs), they should not be changed if one wishes to
reproduce the results from our paper. Thus, to recreate our experiments, one can
use the command

```
bazel run main:federated_trainer -- {optimizer flags}
--experiment_name={experiment name}
```

where the optimizer flags are discussed above. The metrics of the training
procedure are logged and written to the directory `tmp/fed_opt/{experiment
name}`.

While we have attempted to make our results as reproducible as possible by even
setting a seed to recreate which clients were sampled at each round (governed by
`--client_dataset_seed`), we note that randomness in client sampling and
heterogeneity across clients can lead to greater variance than in centralized
machine learning settings. We also note that choices of optimizer
hyperparameters are often vital.

# Federated Optimization

This directory contains source code for evaluating federated learning with
different optimizers on various models and tasks. For a more general look at
using TensorFlow Federated for research, see
[Using TFF for Federated Learning Research](https://www.tensorflow.org/federated/tff_for_research).

Some pip packages are required by this library, and may need to be installed:

```
pip install absl-py
pip install attr
pip install dm-tree
pip install semantic-version
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
dataset), as well as binaries for performing federated and non-federated
(centralized) training. These are in
`optimization/{task}/run_{federated/centralized}.py`.

There is also a `shared` directory with utilities specific to these experiments,
such as implementations of metrics used for evaluation.

### Example usage

Suppose we wish to train a convolutional network on EMNIST for purposes of
character recognition, using federated optimization. This is done in
`emnist/run_federated.py`. Various aspects of the federated training procedure
can be customized via absl flags. For example, from this directory one could
run:

```
bazel run emnist:run_federated -- --total_rounds=100
--client_optimizer=sgd --client_learning_rate=0.1 --server_optimizer=sgd
--server_learning_rate=1.0 --clients_per_round=10 --client_epochs_per_round=1
--experiment_name=emnist_fedavg_experiment
```

This will run 100 communication rounds of federated training, using SGD on both
the client and server, with learning rates of 0.1 and 1.0 respectively. The
experiment uses 10 clients in each round, and performs 1 training epoch on each
client's dataset. The `experiment_name` flag is for the purposes of writing
metrics.

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

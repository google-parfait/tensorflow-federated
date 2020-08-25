# Adaptive Learning Rate Decay

This directory contains libraries for performing federated learning with
adaptive learning rate decay. For a more general look at using TensorFlow
Federated for research, see
[Using TFF for Federated Learning Research](https://www.tensorflow.org/federated/tff_for_research).
This directory contains a more advanced version of federated averaging, and
assumes some familiarity with libraries such as
[tff.learning.build_federated_averaging_process](https://www.tensorflow.org/federated/api_docs/python/tff/learning/build_federated_averaging_process).

## Dependencies

To use this library, one should first follow the instructions
[here](https://github.com/tensorflow/federated/blob/master/docs/install.md) to
install TensorFlow Federated using pip. Other pip packages are required by this
library, and may need to be installed. They can be installed via the following
commands:

```
pip install absl-py
pip install attr
pip install dm-tree
pip install numpy
pip install pandas
pip install tensorflow
```

## General description

This example contains two main libraries,
[adaptive_fed_avg.py](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/adaptive_lr_decay/adaptive_fed_avg.py)
and
[callbacks.py](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/adaptive_lr_decay/callbacks.py).
The latter implements learning rate callbacks that adaptively decay learning
rates based on moving averages of metrics. This is relevant in the federated
setting, as we may wish to decay learning rates based on the average training
loss across rounds.

These callbacks are used in `adaptive_fed_avg.py` to perform federated averaging
with adaptive learning rate decay. Notably, `adaptive_fed_avg.py` decouples
client and server leaerning rates so that they can be decayed independently, and
so that we do not conflate their effects. In order to do this adaptive decay,
the iterative process computes metrics before and during training. The metrics
computed before training are used to inform the learning rate decay throughout.

## Example usage

Suppose we wanted to run a training process in which we decay the client
learning rate when the training loss plateaus. We would first create a client
learning rate callback via a command such as

```
client_lr_callback = callbacks.create_reduce_lr_on_plateau(
  learning_rate=0.5,
  decay_factor=0.1,
  monitor='loss')
```

Every time a new loss value `loss_value` is computed, you can call
`callbacks.update_reduce_lr_on_plateau(client_lr_callback, loss_value)` This
will update the moving average of loss maintained by the callback, as well as
the smallest loss value seen so far. If the loss is deemed to have plateaued
according to these metrics, the client learning rate will be decayed by a factor
of `client_lr_callback.decay_factor`.

These callbacks are incorporated into `adaptive_fed_avg` so that the learning
rate (client and/or server) will be decayed automatically as learning
progresses. For example, suppose we do not want the server LR to decay. Then we
can construct `server_lr_callback =
callbacks.create_reduce_lr_on_plateau(learning_rate=1.0, decay_factor=1.0)`, and
then, using these callbacks with a `model_fn` that returns an uncompiled
[tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model), we
can call

<!-- mdformat off(This code snippet is sensitive to automatic formatting changes) -->
```
iterative_process = adaptive_fed_avg.build_fed_avg_process(
  model_fn,
  client_lr_callback,
  server_lr_callback,
  client_optimizer_fn=tf.keras.optimizers.SGD,
  server_optimizer_fn=tf.keras.optimizers.SGD)
```
<!-- mdformat on -->

This will build an iterative process that trains the model created by `model_fn`
using federated averaging, decaying the client learning rate as training
progresses according to whether the loss plateaus.

## More detailed usage

The learning rate callbacks have many other configurations that may improve
performance. For example, you can set a `cooldown` period (preventing the
learning rate from decaying for a number of rounds after it has decayed), or
configure how many consecutive rounds of plateauing loss must be observed before
decaying the learning rate (via the `patience` argument). For more details, see
the documentation for
[callbacks.py](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/adaptive_lr_decay/callbacks.py).

## Benchmarking experiments

For the purposes of testing out our adaptive learning rate decay methods, we
have also included federated training binaries for four different datasets,
CIFAR-100, FEMNIST, Shakespeare, and Stack Overflow. For Stack Overflow, we
perform two distinct tasks, tag prediction, and next word prediction. A summary
of the datasets, models, and tasks are given below.

Dataset        | Model                             | Task Summary              | Binary
-------------- | --------------------------------- | ------------------------- | ------
CIFAR-100      | ResNet-18 (with GroupNorm layers) | Image classification      | [run_federated_cifar100.py](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/adaptive_lr_decay/run_federated_cifar100.py)
FEMNIST        | Convolutional Neural Network      | Digit recognition         | [run_federated_emnist.py](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/adaptive_lr_decay/run_federated_emnist.py)
Shakespeare    | RNN with 2 LSTM layers            | Next character prediction | [run_federated_shakespeare.py](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/adaptive_lr_decay/run_federated_shakespeare.py)
Stack Overflow | RNN with 1 LSTM layer             | Next word prediction      | [run_federated_stackoverflow.py](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/adaptive_lr_decay/run_federated_stackoverflow.py)
Stack Overflow | Logistic regression classifier    | Tag prediction            | [run_federated_stackoverflow_lr.py](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/adaptive_lr_decay/run_federated_stackoverflow_lr.py)

To run the corresponding binaries, we require [Bazel](https://www.bazel.build/).
Instructions for installing Bazel can be found
[here](https://docs.bazel.build/versions/master/install.html).

To run a baseline classifier on CIFAR-100, for example, one would run (inside
this directory):

```
bazel run :federated_trainer -- --task=cifar100 --total_rounds=100
--client_optimizer=sgd --client_learning_rate=0.1 --server_optimizer=sgd
--server_learning_rate=0.1 --clients_per_round=10 --client_epochs_per_round=1
--experiment_name=cifar100_classification
```

This will run 100 communication rounds of federated averaging, using SGD on both
the server and client, with 10 clients per round and 1 client epoch per round.
If instead you wanted each client to perform 10 gradient steps (irrespective of
the size of their local dataset), you could set `--client_epochs_per_round=-1
--max_batches_per_client=10`. For more details on these flags, see
[federated_trainer.py](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/adaptive_lr_decay/federated_trainer.py).

In the example above, the client and server both use learning rates of 0.1. By
default, when the loss plateaus, the iterative process constructed will
adaptively decay the client learning rate by a factor of 0.1 and the server
learning rate by a factor of 0.9. To customize the adaptive learning rate decay
further, one could alter the server learning rate decay factor, the window size
used to estimate the global loss, the minimum learning rate, and other
configurations. These are configured via abseil flags. For a list of flags
configuring the adaptive learning rate decay, see
[federated_trainer.py](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/adaptive_lr_decay/federated_trainer.py).

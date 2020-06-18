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

This example contains two main libraries, `adaptive_fed_avg.py` and
`callbacks.py`. The latter implements learning rate callbacks that adaptively
decay learning rates based on moving averages of metrics. This is relevant in
the federated setting, as we may wish to decay learning rates based on the
average training loss across rounds.

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
can construct `server_lr_callback = callbacks.create_reduce_lr_on_plateau(
learning_rate=1.0, decay_factor=1.0)` and then using these callbacks with some
`model_fn`, we can call

<!-- mdformat off(This code snippet is sensitive to automatic formatting changes) -->
```
iterative_process = adaptive_fed_avg.build_fed_avg_process(
  model_fn,
  client_lr_callback,
  callbacks.update_reduce_lr_on_plateau,
  server_lr_callback,
  callbacks.update_reduce_lr_on_plateau,
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
the documentation for `callbacks.py`.

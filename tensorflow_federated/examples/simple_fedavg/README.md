# Minimal Stand-Alone Implementation of Federated Averaging

This is intended to be a flexible and minimal implementation of Federated
Averaging, and the code is designed to be modular and re-usable. This
implmentation of the federated averaging algorithm only uses key TFF functions
and does not depend on advanced features in `tff.learning`. See
[federated_averaging.py](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/python/learning/federated_averaging.py)
for a more full-featured implementation.

## Instructions

A minimal implementation of the
[Federated Averaging](https://arxiv.org/abs/1602.05629) algorithm is provided
[here](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg),
along with an example
[federated EMNIST experiment](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py).
The implementation demonstrates the three main types of logic of a typical
federated learning simulation.

*   An outer python script that drives the simulation by selecting simulated
    clients from a dataset and then executing federated learning algorithms.

*   Individual pieces of TensorFlow code that run in a single location (e.g., on
    clients or on a server). The code pieces are typically `tf.function`s that
    can be used and tested outside TFF.

*   The orchestration logic binds together the local computations by wrapping
    them as `tff.tf_computation`s and using key TFF functions like
    `tff.federated_broadcast` and `tff.federated_map` inside a
    `tff.federated_computation`.

This EMNIST example can easily be adapted for experimental changes:

*   In the driver file
    [emnist_fedavg_main](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py),
    we can change the
    [dataset](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py#L49-L79),
    the
    [neural network architecture](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py#L82-L122),
    the
    [server_optimizer](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py#L125-L126),
    and the
    [client_optimizer](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py#L129-L130)
    for cutomized applications. Note that we need a
    [model wrapper](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py#L151-152),
    and build an
    [iterative process](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py#L154-L155)
    with TFF. We define a stand-alone model wrapper for keras models in
    [simple_fedavg_tf](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L39-L81),
    which can be substituted with `tff.learning.Model` by calling
    `tff.learning.from_keras_model`. Note that the inner
    [keras_model](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py#L174)
    of `tff.learning.Model` may not be directly accessible for evaluation.

*   In the TF function file
    [simple_fedavg_tf](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py),
    we have more control over the local computations performed in optimization
    process. In each round, on the server side, we will update the
    [ServerState](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L102-L113)
    in
    [server_update](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L131-L141)
    function; we then build a
    [BroadcastMessage](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L116-L128)
    with the
    [build_server_broadcast_message](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L165-L181)
    function to prepare for broadcasting from server to clients; on the client
    side, we perform local updates with the
    [client_update](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L184-L222)
    function and return
    [ClientOutput](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L84-L99)
    to be sent back to server. Note that server_optimizer defined in
    [emnist_fedavg_main](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py#L125-L126)
    is used in
    [server_update](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L131-L141)
    function; client_optimizer defined in
    [emnist_fedavg_main](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py#L129-L130)
    is used in
    [client_update](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py#L184-L222).
    These functions are used as local computation building blocks in the overall
    TFF computation, which handles the broadcasting and aggregation between
    server and clients.

*   In the TFF file
    [simple_fedavg_tff](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py),
    we have control over the orchestration strategy. We take the
    [weighted average](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py#L132-L133)
    of client updates to update the model kept in server state. More detailed
    instruction on the usage of TFF functions `federated_broadcast`,
    `federated_map`, and `federated_mean` can be found in the
    [tutorial](https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_1).

We expect researchers working on applications and models only need to change the
driver file
[emnist_fedavg_main](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py),
researchers working on optimization may implement most of the ideas by writing
pure TF code in the TF file
[simple_fedavg_tf](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py),
while researchers who need more control over the orchestration strategy may get
familiar with TFF code in
[simple_fedavg_tff](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py).
We encourage readers to consider the following exercises for using this set of
code for your research:

1.  Try a more complicated server optimizer such as ADAM. You only need to
    change
    [emnist_fedavg_main](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py).

1.  Implement a model that uses L2 regularization. You will need to change the
    model definition in
    [emnist_fedavg_main](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/emnist_fedavg_main.py)
    and add Keras regularization losses in the `KerasModelWrapper` class in
    [simple_fedavg_tf](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py).

1.  Implement a decaying learning rate schedule on the clients based on the
    global round, using the `round_num` broadcasted to the clients in
    [simple_fedavg_tf](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tf.py).

1.  Implement a more complicated aggregation procedure that drops the client
    updates with the largest and smallest l2 norms. You will need to change
    [simple_fedavg_tff](https://github.com/tensorflow/federated/blob/main/tensorflow_federated/examples/simple_fedavg/simple_fedavg_tff.py).

## Citation

```
@inproceedings{mcmahan2017communication,
  title={Communication-Efficient Learning of Deep Networks from
  Decentralized Data},
  author={McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson,
  Seth and y Arcas, Blaise Aguera},
  booktitle={Artificial Intelligence and Statistics},
  pages={1273--1282},
  year={2017}
  }
```

# Minimal Stand-Alone Implementation of Federated Averaging

This is intended to be a flexible and minimal implementation of Federated
Averaging, and the code is designed to be modular and reusable. This
implementation of the federated averaging algorithm only uses key TFF functions
and does not depend on advanced features in `tff.learning`. See
[fed_avg_.py](https://github.com/google-parfait/tensorflow-federated/blob/main/tensorflow_federated/python/learning/algorithms/fed_avg.py)
for a more full-featured implementation.

## Instructions

A minimal implementation of the
[Federated Averaging](https://arxiv.org/abs/1602.05629) algorithm is provided
[here](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg),
along with an example
[federated EMNIST experiment](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/emnist_fedavg_main.py).
The implementation demonstrates the three main types of logic of a typical
federated learning simulation.

*   An outer python script that drives the simulation by selecting simulated
    clients from a dataset and then executing federated learning algorithms.

*   Individual pieces of TensorFlow code that run in a single location (e.g., on
    clients or on a server). The code pieces are typically `tf.function`s that
    can be used and tested outside TFF.

*   The orchestration logic binds together the local computations by wrapping
    them as `tff.tensorflow.computation`s and using key TFF functions like
    `federated_language.federated_broadcast` and
    `federated_language.federated_map` inside a
    `federated_language.federated_computation`.

This EMNIST example can easily be adapted for experimental changes:

*   In the driver file
    [emnist_fedavg_main](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/emnist_fedavg_main.py),
    we can change the
    [dataset](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/emnist_fedavg_main.py),
    the
    [neural network architecture](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/emnist_fedavg_main.py),
    the
    [server_optimizer](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/emnist_fedavg_main.py),
    and the
    [client_optimizer](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/emnist_fedavg_main.py)
    for customized applications. Note that we need a
    [model wrapper](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/emnist_fedavg_main.py),
    and build an
    [iterative process](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/emnist_fedavg_main.py)
    with TFF. We construct a `tff.learning.models.FunctionalModel` from a Keras
    model by calling `tff.learning.models.functional_model_from_keras`. Note
    that the inner
    [keras_model](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/emnist_fedavg_main.py)
    of `tff.learning.models.FunctionalModel` is not directly accessible for
    evaluation.

*   In the TF function file
    [simple_fedavg_tf](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/simple_fedavg_tf.py),
    we have more control over the local computations performed in optimization
    process. In each round, on the server side, we will update the
    [ServerState](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/simple_fedavg_tf.py)
    in
    [server_update](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/simple_fedavg_tf.py)
    function; we then build a
    [BroadcastMessage](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/simple_fedavg_tf.py)
    with the
    [build_server_broadcast_message](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/simple_fedavg_tf.py)
    function to prepare for broadcasting from server to clients; on the client
    side, we perform local updates with the
    [client_update](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/simple_fedavg_tf.py)
    function and return
    [ClientOutput](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/simple_fedavg_tf.py)
    to be sent back to server. Note that server_optimizer defined in
    [emnist_fedavg_main](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/emnist_fedavg_main.py)
    is used in
    [server_update](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/simple_fedavg_tf.py)
    function; client_optimizer defined in
    [emnist_fedavg_main](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/emnist_fedavg_main.py)
    is used in
    [client_update](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/simple_fedavg_tf.py).
    These functions are used as local computation building blocks in the overall
    TFF computation, which handles the broadcasting and aggregation between
    server and clients.

*   In the TFF file
    [simple_fedavg_tff](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/simple_fedavg_tff.py),
    we have control over the orchestration strategy. We take the
    [weighted average](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/simple_fedavg_tff.py)
    of client updates to update the model kept in server state. More detailed
    instruction on the usage of TFF functions `federated_broadcast`,
    `federated_map`, and `federated_mean` can be found in the
    [tutorial](https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_1).

We expect researchers working on applications and models only need to change the
driver file
[emnist_fedavg_main](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/emnist_fedavg_main.py),
researchers working on optimization may implement most of the ideas by writing
pure TF code in the TF file
[simple_fedavg_tf](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/simple_fedavg_tf.py),
while researchers who need more control over the orchestration strategy may get
familiar with TFF code in
[simple_fedavg_tff](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/simple_fedavg_tff.py).
We encourage readers to consider the following exercises for using this set of
code for your research:

1.  Try a more complicated server optimizer such as ADAM. You only need to
    change
    [emnist_fedavg_main](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/emnist_fedavg_main.py).

1.  Implement a model that uses L2 regularization. You will need to change the
    model definition in
    [emnist_fedavg_main](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/emnist_fedavg_main.py)
    and add regularization losses to the loss computation in
    [simple_fedavg_tf](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/simple_fedavg_tf.py).

1.  Implement a decaying learning rate schedule on the clients based on the
    global round, using the `round_num` broadcasted to the clients in
    [simple_fedavg_tf](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/simple_fedavg_tf.py).

1.  Implement a more complicated aggregation procedure that drops the client
    updates with the largest and smallest l2 norms. You will need to change
    [simple_fedavg_tff](https://github.com/google-parfait/tensorflow-federated/blob/main/examples/simple_fedavg/simple_fedavg_tff.py).

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

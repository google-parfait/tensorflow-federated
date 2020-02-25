# Minimal Stand-Alone Implementation of Federated Averaging

This is intended to be a flexible and minimal implementation of Federated
Averaging, and the code is designed to be modular and re-usable. See
[federated_averaging.py](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/learning/federated_averaging.py)
for a more full-featured implementation.

## Instructions

A minimal implementation of the
[Federated Averaging](https://arxiv.org/abs/1602.05629) algorithm is provided
[here](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg),
along with an example
[federated EMNIST experiment](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py).
This example can easily be adapted for experimental changes:

*   In the driver file
    [emnist_fedavg_main](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py),
    we can change the
    [dataset](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py#L49-L78),
    the
    [neural network architecture](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py#L81-L118),
    the
    [server_optimizer](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py#L123-L124),
    and the
    [client_optimizer](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py#L127-L128)
    for cutomized applications. Note that we need a
    [model wrapper](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py#L158-L163),
    and build an
    [iterative process](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py#L165-L166)
    with TFF.

*   In the TF function file
    [simple_fedavg_tf](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tf.py),
    we have more control over the local computations performed in optimization
    process. In each round, on the server side, we will update the
    [ServerState](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tf.py#L122-L133)
    in
    [server_update](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tf.py#L151-L182)
    function; we then build a
    [BroadcastMessage](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tf.py#L136-L148)
    with the
    [build_server_broadcast_message](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tf.py#L185-L201)
    function to prepare for broadcasting from server to clients; on the client
    side, we perform local updates with the
    [client_update](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tf.py#L204-L242)
    function and return
    [ClientOutput](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tf.py#L104-L119)
    to be sent back to server. Note that server_optimizer defined in
    [emnist_fedavg_main](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py#L123-L124)
    is used in
    [server_update](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tf.py#L151-L182)
    function; client_optimizer defined in
    [emnist_fedavg_main](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py#L127-L128)
    is used in
    [client_update](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tf.py#L204-L242).
    These functions are used as local computation building blocks in the overall
    TFF computation, which handles the broadcasting and aggregation between
    server and clients.

*   In the TFF file
    [simple_fedavg_tff](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tff.py),
    we have control over the orchestration strategy. We take the
    [weighted average](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tff.py#L131-L132)
    of client updates to update the model kept in server state. More detailed
    instruction on the usage of TFF functions `federated_broadcast`,
    `federated_map`, and `federated_mean` can be found in the
    [tutorial](https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_1).

We expect researchers working on applications and models only need to change the
driver file
[emnist_fedavg_main](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py),
researchers working on optimization may implement most of the ideas by writing
pure TF code in the TF file
[simple_fedavg_tf](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tf.py),
while researchers who need more control over the orchestration strategy may get
familiar with TFF code in
[simple_fedavg_tff](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tff.py).
We encourage readers to consider the following exercises for using this set of
code for your research:

1.  Try a more complicated server optimizer such as ADAM. You only need to
    change
    [emnist_fedavg_main](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/emnist_fedavg_main.py).

1.  Implement a decaying learning rate schedule on the clients based on the
    global round, using the `round_num` broadcasted to the clients in
    [simple_fedavg_tf](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tf.py).

1.  Implement a more complicated aggregation procedure that drops the client
    updates with the largest and smallest l2 norms. You will need to change
    [simple_fedavg_tff](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/research/simple_fedavg/simple_fedavg_tff.py).

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

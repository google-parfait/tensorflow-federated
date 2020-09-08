# Demo of Stateful Clients in Federated Averaging

This is an example project demonstrating how to implment stateful clients in TFF
simulation. Note that client states are generally discouraged in cross-device
federated learning (see definition in
[Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977))
because of the large number of total clients and the intention for privacy
protection.

This project is based on the standalone implementaion of Federated Averaging
algorithm in
[`simple_fedavg`](https://github.com/tensorflow/federated/blob/master/tensorflow_federated/python/examples/simple_fedavg).
We introduce a coutner on each client, which tracks the total number of
iterations for model training on the clients. For example, if client A has been
sampled m times at round n, and each time local model training has been run with
q iterations, then the counter in client A state would be q\*m at round n. On
the server, we also aggregate the total number of iterations for all the clients
sampled in this round.

The client states are stored in memory in the current implementation. It would
become challenging if the size of as single client state and/or the number of
clients become too large. A possible extension is to use high performance IO to
efficiently save and load client states on disk. We would encourge readers to
explore and contribute.

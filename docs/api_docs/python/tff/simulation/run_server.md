<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.run_server" />
<meta itemprop="path" content="Stable" />
</div>

# tff.simulation.run_server

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/server_utils.py">View
source</a>

Runs a gRPC server hosting a simulation component in this process.

```python
tff.simulation.run_server(
    executor,
    num_threads,
    port,
    credentials=None,
    options=None
)
```

<!-- Placeholder for "Used in" -->

The server runs indefinitely, but can be stopped by a keyboard interrrupt.

#### Args:

*   <b>`executor`</b>: The executor to be hosted by the server.
*   <b>`num_threads`</b>: The number of network threads to use for handling gRPC
    calls.
*   <b>`port`</b>: The port to listen on (for gRPC), must be a non-zero integer.
*   <b>`credentials`</b>: The optional credentials to use for the secure
    connection if any, or `None` if the server should open an insecure port. If
    specified, must be a valid `ServerCredentials` object that can be accepted
    by the gRPC server's `add_secure_port()`.
*   <b>`options`</b>: The optional `list` of server options, each in the `(key,
    value)` format accepted by the `grpc.server()` constructor.

#### Raises:

*   <b>`ValueError`</b>: If `num_threads` or `port` are invalid.

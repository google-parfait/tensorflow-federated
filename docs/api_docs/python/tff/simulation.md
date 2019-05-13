<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tff.simulation

The public API for experimenters running federated learning simulations.

Defined in
[`simulation/__init__.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/__init__.py).

<!-- Placeholder for "Used in" -->

## Modules

[`datasets`](../tff/simulation/datasets.md) module: Datasets for running
Federated Learning experiments in simulation.

## Classes

[`class ClientData`](../tff/simulation/ClientData.md): Object to hold a dataset
and a mapping of clients to examples.

[`class FilePerUserClientData`](../tff/simulation/FilePerUserClientData.md): A
`tf.simulation.ClientData` that maps a set of files to a dataset.

[`class FromTensorSlicesClientData`](../tff/simulation/FromTensorSlicesClientData.md):
ClientData based on `tf.data.Dataset.from_tensor_slices`.

[`class HDF5ClientData`](../tff/simulation/HDF5ClientData.md): A
<a href="../tff/simulation/ClientData.md"><code>tff.simulation.ClientData</code></a>
backed by an HDF5 file.

[`class TransformingClientData`](../tff/simulation/TransformingClientData.md):
Transforms client data, potentially expanding by adding pseudo-clients.

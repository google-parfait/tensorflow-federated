<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tff.simulation

Defined in
[`simulation/__init__.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/__init__.py).

The public API for experimenters running federated learning simulations.

## Modules

[`datasets`](../tff/simulation/datasets.md) module: Datasets for running
Federated Learning experiments in simulation.

## Classes

[`class ClientData`](../tff/simulation/ClientData.md): Object to hold a dataset
and a mapping of clients to examples.

[`class FilePerUserClientData`](../tff/simulation/FilePerUserClientData.md): A
`tf.simulation.ClientData` that maps a set of files to a dataset.

[`class HDF5ClientData`](../tff/simulation/HDF5ClientData.md): A
`tf.simulation.ClientData` backed by an HDF5 file.

[`class TransformingClientData`](../tff/simulation/TransformingClientData.md):
Expands client data by performing transformations.

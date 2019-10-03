<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.datasets" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tff.simulation.datasets

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/datasets/__init__.py">View
source</a>

Datasets for running Federated Learning experiments in simulation.

<!-- Placeholder for "Used in" -->

## Modules

[`emnist`](../../tff/simulation/datasets/emnist.md) module: Libraries for the
federated EMNIST dataset for simulation.

[`shakespeare`](../../tff/simulation/datasets/shakespeare.md) module: Libraries
for the Shakespeare dataset for federated learning simulation.

[`stackoverflow`](../../tff/simulation/datasets/stackoverflow.md) module:
Libraries for the Stackoverflow dataset for federated learning simulation.

## Functions

[`build_dataset_mixture(...)`](../../tff/simulation/datasets/build_dataset_mixture.md):
Build a new dataset that probabilistically returns examples.

[`build_single_label_dataset(...)`](../../tff/simulation/datasets/build_single_label_dataset.md):
Build a new dataset that only yields examples with a particular label.

[`build_synthethic_iid_datasets(...)`](../../tff/simulation/datasets/build_synthethic_iid_datasets.md):
Constructs an iterable of IID clients from a `tf.data.Dataset`.

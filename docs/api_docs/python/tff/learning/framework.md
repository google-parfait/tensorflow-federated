<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.framework" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tff.learning.framework

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/framework/__init__.py">View
source</a>

The public API for contributors who develop federated learning algorithms.

<!-- Placeholder for "Used in" -->

## Classes

[`class ClientDeltaFn`](../../tff/learning/framework/ClientDeltaFn.md):
Represents a client computation that produces an update to a model.

[`class ClientOutput`](../../tff/learning/framework/ClientOutput.md): Structure
for outputs returned from clients during federated optimization.

[`class EnhancedModel`](../../tff/learning/framework/EnhancedModel.md): A
wrapper around a Model that adds sanity checking and metadata helpers.

[`class EnhancedTrainableModel`](../../tff/learning/framework/EnhancedTrainableModel.md):
A wrapper around a Model that adds sanity checking and metadata helpers.

[`class ModelWeights`](../../tff/learning/framework/ModelWeights.md): A
container for the trainable and non-trainable variables of a `Model`.

## Functions

[`build_model_delta_optimizer_process(...)`](../../tff/learning/framework/build_model_delta_optimizer_process.md):
Constructs
<a href="../../tff/utils/IterativeProcess.md"><code>tff.utils.IterativeProcess</code></a>
for Federated Averaging or SGD.

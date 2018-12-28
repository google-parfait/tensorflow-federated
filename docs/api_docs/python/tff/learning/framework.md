<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.framework" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tff.learning.framework

The public API for contributors who develop federated learning algorithms.

## Classes

[`class ClientDeltaFn`](../../tff/learning/framework/ClientDeltaFn.md): Represents a client computation that produces an update to a model.

[`class ClientOutput`](../../tff/learning/framework/ClientOutput.md): ClientOutput(weights_delta, weights_delta_weight, model_output, optimizer_output)

[`class EnhancedModel`](../../tff/learning/framework/EnhancedModel.md): A wrapper around a Model that adds sanity checking and metadata helpers.

[`class EnhancedTrainableModel`](../../tff/learning/framework/EnhancedTrainableModel.md): A wrapper around a Model that adds sanity checking and metadata helpers.

[`class ModelWeights`](../../tff/learning/framework/ModelWeights.md): A container for the trainable and non-trainable variables of a `Model`.

[`class SequentialTffComputation`](../../tff/learning/framework/SequentialTffComputation.md): Container for a pair of TFF computations defining sequential processing.

## Functions

[`build_model_delta_optimizer_tff(...)`](../../tff/learning/framework/build_model_delta_optimizer_tff.md): Constructs complete TFF computations for Federated Averaging or SGD.


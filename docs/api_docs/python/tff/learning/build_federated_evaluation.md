<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.build_federated_evaluation" />
<meta itemprop="path" content="Stable" />
</div>

# tff.learning.build_federated_evaluation

```python
tff.learning.build_federated_evaluation(model_fn)
```

Defined in
[`learning/federated_evaluation.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/federated_evaluation.py).

<!-- Placeholder for "Used in" -->

Builds the TFF computation for federated evaluation of the given model.

#### Args:

*   <b>`model_fn`</b>: A no-argument function that returns a
    <a href="../../tff/learning/Model.md"><code>tff.learning.Model</code></a>.

#### Returns:

A federated computation (an instance of
<a href="../../tff/Computation.md"><code>tff.Computation</code></a>) that
accepts model parameters and federated data, and returns the evaluation metrics
as aggregated by
<a href="../../tff/learning/Model.md#federated_output_computation"><code>tff.learning.Model.federated_output_computation</code></a>.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.framework.enhance" />
<meta itemprop="path" content="Stable" />
</div>

# tff.learning.framework.enhance

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/model_utils.py">View
source</a>

Wraps a
<a href="../../../tff/learning/Model.md"><code>tff.learning.Model</code></a> as
an `EnhancedModel`.

```python
tff.learning.framework.enhance(model)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`model`</b>: A
    <a href="../../../tff/learning/Model.md"><code>tff.learning.Model</code></a>.

#### Returns:

An `EnhancedModel` or `TrainableEnhancedModel`, depending on the type of the
input model. If `model` has already been wrapped as such, this is a no-op.

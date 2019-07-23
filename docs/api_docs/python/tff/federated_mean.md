<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_mean" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_mean

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/intrinsics.py">View
source</a>

Computes a <a href="../tff.md#SERVER"><code>tff.SERVER</code></a> mean of
`value` placed on <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

```python
tff.federated_mean(
    value,
    weight=None
)
```

### Used in the tutorials:

*   [Custom Federated Algorithms, Part 1: Introduction to the Federated Core](https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_1)
*   [Custom Federated Algorithms, Part 2: Implementing Federated Averaging](https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_2)
*   [Federated Learning for Image Classification](https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification)

#### Args:

*   <b>`value`</b>: The value of which the mean is to be computed. Must be of a
    TFF federated type placed at
    <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>. The value may be
    structured, e.g., its member constituents can be named tuples. The tensor
    types that the value is composed of must be floating-point or complex.
*   <b>`weight`</b>: An optional weight, a TFF federated integer or
    floating-point tensor value, also placed at
    <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

#### Returns:

A representation at the <a href="../tff.md#SERVER"><code>tff.SERVER</code></a>
of the mean of the member constituents of `value`, optionally weighted with
`weight` if specified (otherwise, the member constituents contributed by all
clients are equally weighted).

#### Raises:

*   <b>`TypeError`</b>: if `value` is not a federated TFF value placed at
    <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>, or if `weight` is
    not a federated integer or a floating-point tensor with the matching
    placement.

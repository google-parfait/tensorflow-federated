<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.sequence_sum" />
<meta itemprop="path" content="Stable" />
</div>

# tff.sequence_sum

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/intrinsics.py">View
source</a>

Computes a sum of elements in a sequence.

```python
tff.sequence_sum(value)
```

### Used in the tutorials:

*   [Custom Federated Algorithms, Part 2: Implementing Federated Averaging](https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_2)

#### Args:

*   <b>`value`</b>: A value of a TFF type that is either a sequence, or a
    federated sequence.

#### Returns:

The sum of elements in the sequence. If the argument `value` is of a federated
type, the result is also of a federated type, with the sum computed locally and
independently at each location (see also a discussion on `sequence_map` and
`sequence_reduce`).

#### Raises:

*   <b>`TypeError`</b>: If the arguments are of wrong or unsupported types.

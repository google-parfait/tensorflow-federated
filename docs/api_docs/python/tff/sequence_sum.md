<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.sequence_sum" />
<meta itemprop="path" content="Stable" />
</div>

# tff.sequence_sum

``` python
tff.sequence_sum(value)
```



Defined in [`core/api/intrinsics.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/intrinsics.py).

<!-- Placeholder for "Used in" -->

Computes a sum of elements in a sequence.

#### Args:

* <b>`value`</b>: A value of a TFF type that is either a sequence, or a federated
    sequence.


#### Returns:

The sum of elements in the sequence. If the argument `value` is of a
federated type, the result is also of a federated type, with the sum
computed locally and independently at each location (see also a discussion
on `sequence_map` and `sequence_reduce`).


#### Raises:

* <b>`TypeError`</b>: If the arguments are of wrong or unsupported types.
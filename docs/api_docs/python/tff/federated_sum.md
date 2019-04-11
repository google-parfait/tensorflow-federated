<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_sum" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_sum

``` python
tff.federated_sum(value)
```



Defined in [`core/api/intrinsics.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/intrinsics.py).

<!-- Placeholder for "Used in" -->

Computes a sum at <a href="../tff.md#SERVER"><code>tff.SERVER</code></a> of a `value` placed on the <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

#### Args:

* <b>`value`</b>: A value of a TFF federated type placed at the <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.


#### Returns:

A representation of the sum of the member constituents of `value` placed
on the <a href="../tff.md#SERVER"><code>tff.SERVER</code></a>.


#### Raises:

* <b>`TypeError`</b>: if the argument is not a federated TFF value placed at
    <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.
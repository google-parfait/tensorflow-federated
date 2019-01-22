<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_sum" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_sum

``` python
tff.federated_sum(value)
```



Defined in [`core/api/intrinsics.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/intrinsics.py).

Computes a sum at `SERVER` of a federated value placed on the `CLIENTS`.

#### Args:

* <b>`value`</b>: A value of a TFF federated type placed at the `CLIENTS`.


#### Returns:

A representation of the sum of the member constituents of `value` placed
on the `SERVER`.


#### Raises:

* <b>`TypeError`</b>: if the argument is not a federated TFF value placed at `CLIENTS`.
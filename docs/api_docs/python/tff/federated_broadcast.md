<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_broadcast" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_broadcast

``` python
tff.federated_broadcast(value)
```

Broadcasts a federated value from the `SERVER` to the `CLIENTS`.

#### Args:

* <b>`value`</b>: A value of a TFF federated type placed at the `SERVER`, all members
    of which are equal (the `all_equal` property of the federated type of
   `value` is True).


#### Returns:

A representation of the result of broadcasting: a value of a TFF federated
type placed at the `CLIENTS`, all members of which are equal.


#### Raises:

* <b>`TypeError`</b>: if the argument is not a federated TFF value placed at the
    `SERVER`.
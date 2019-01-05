<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_zip" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_zip

``` python
tff.federated_zip(value)
```

Converts an N-tuple of federated values into a federated N-tuple value.

#### Args:

* <b>`value`</b>: A value of a TFF named tuple type, the elements of which are
    federated values placed at the `CLIENTS`.


#### Returns:

A federated value placed at the `CLIENTS` in which every member component
at the given client is a named tuple that consists of the corresponding
 member components of the elements of `value` residing at that client.


#### Raises:

* <b>`TypeError`</b>: if the argument is not a named tuple of federated values placed
  at 'CLIENTS`.
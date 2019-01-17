<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.Computation" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="type_signature"/>
<meta itemprop="property" content="__call__"/>
</div>

# tff.Computation

## Class `Computation`

Defined in
[`core/api/computation_base.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/computation_base.py).

An abstract interface for all classes that represent computations.

## Properties

<h3 id="type_signature"><code>type_signature</code></h3>

Provides access to the type signature of this computation.

#### Returns:

An instance of a class that represents this computation's type signature.

## Methods

<h3 id="__call__"><code>__call__</code></h3>

```python
__call__(
    *args,
    **kwargs
)
```

Invokes the computation with the given arguments in the given context.

#### Args:

*   <b>`*args`</b>: The positional arguments.
*   <b>`**kwargs`</b>: The keyword-based arguments.

#### Returns:

The result of invoking the computation, the exact form of which depends on the
context.

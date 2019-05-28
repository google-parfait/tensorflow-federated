<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.Value" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="type_signature"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
</div>

# tff.Value

## Class `Value`

An abstract base class for all values in the bodies of TFF computations.

Inherits From: [`TypedObject`](../tff/TypedObject.md)

Defined in
[`core/api/value_base.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/value_base.py).

<!-- Placeholder for "Used in" -->

This interface is only relevant in the context of non-TensorFlow computations,
such as those that represent federated orchestration logic. The bodies of such
computations will contain a mixture of federated communication operators, and
calls to TensorFlow computations embedded in them as subcomponents. All values
that appear in those computations implement this common interface, just like all
values in TensorFlow computations appear as tensors.

Outside of the bodies of composite non-TensorFlow computations, this interface
is not used. All fully constructed computations implement
<a href="../tff/Computation.md"><code>tff.Computation</code></a>.

## Properties

<h3 id="type_signature"><code>type_signature</code></h3>

Returns the TFF type of this object (an instance of
<a href="../tff/Type.md"><code>tff.Type</code></a>).

## Methods

<h3 id="__call__"><code>__call__</code></h3>

```python
__call__(
    *args,
    **kwargs
)
```

For values of functional types, invokes this value on given arguments.

<h3 id="__getitem__"><code>__getitem__</code></h3>

```python
__getitem__(index)
```

For values of a named tuple type, returns the element at `index`.

<h3 id="__iter__"><code>__iter__</code></h3>

```python
__iter__()
```

For values of a named tuple type, iterates over the tuple elements.

<h3 id="__len__"><code>__len__</code></h3>

```python
__len__()
```

For values of a named tuple type, returns the number of elements.

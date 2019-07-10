<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.Call" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="argument"/>
<meta itemprop="property" content="function"/>
<meta itemprop="property" content="proto"/>
<meta itemprop="property" content="type_signature"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_proto"/>
</div>

# tff.framework.Call

## Class `Call`

A representation of a function invocation in TFF's internal language.

Inherits From:
[`ComputationBuildingBlock`](../../tff/framework/ComputationBuildingBlock.md)

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py>View
source</a>

<!-- Placeholder for "Used in" -->

The call construct takes an argument tuple with two elements, the first being
the function to invoke (represented as a computation with a functional result
type), and the second being the argument to feed to that function. Typically,
the function is either a TFF instrinsic, or a lambda expression.

The concise notation for calls is `foo(bar)`, where `foo` is the function, and
`bar` is the argument.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py>View
source</a>

```python
__init__(
    fn,
    arg=None
)
```

Creates a call to 'fn' with argument 'arg'.

#### Args:

*   <b>`fn`</b>: A value of a functional type that represents the function to
    invoke.
*   <b>`arg`</b>: The optional argument, present iff 'fn' expects one, of a type
    that matches the type of 'fn'.

#### Raises:

*   <b>`TypeError`</b>: if the arguments are of the wrong types.

## Properties

<h3 id="argument"><code>argument</code></h3>

<h3 id="function"><code>function</code></h3>

<h3 id="proto"><code>proto</code></h3>

Returns a serialized form of this object as a pb.Computation instance.

<h3 id="type_signature"><code>type_signature</code></h3>

Returns the TFF type of this object (an instance of
<a href="../../tff/Type.md"><code>tff.Type</code></a>).

## Methods

<h3 id="from_proto"><code>from_proto</code></h3>

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py>View
source</a>

```python
@classmethod
from_proto(
    cls,
    computation_proto
)
```

Returns an instance of a derived class based on 'computation_proto'.

#### Args:

*   <b>`computation_proto`</b>: An instance of pb.Computation.

#### Returns:

An instance of a class that implements 'ComputationBuildingBlock' and that
contains the deserialized logic from in 'computation_proto'.

#### Raises:

*   <b>`NotImplementedError`</b>: if computation_proto contains a kind of
    computation for which deserialization has not been implemented yet.
*   <b>`ValueError`</b>: if deserialization failed due to the argument being
    invalid.

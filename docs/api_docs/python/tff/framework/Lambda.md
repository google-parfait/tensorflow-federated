<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.Lambda" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="parameter_name"/>
<meta itemprop="property" content="parameter_type"/>
<meta itemprop="property" content="proto"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="type_signature"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="compact_representation"/>
<meta itemprop="property" content="formatted_representation"/>
<meta itemprop="property" content="from_proto"/>
<meta itemprop="property" content="structural_representation"/>
</div>

# tff.framework.Lambda

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py">View
source</a>

## Class `Lambda`

A representation of a lambda expression in TFF's internal language.

Inherits From:
[`ComputationBuildingBlock`](../../tff/framework/ComputationBuildingBlock.md)

<!-- Placeholder for "Used in" -->

A lambda expression consists of a string formal parameter name, and a result
expression that can contain references by name to that formal parameter. A
concise notation for lambdas is `(foo -> bar)`, where `foo` is the name of the
formal parameter, and `bar` is the result expression.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py">View
source</a>

```python
__init__(
    parameter_name,
    parameter_type,
    result
)
```

Creates a lambda expression.

#### Args:

*   <b>`parameter_name`</b>: The (string) name of the parameter accepted by the
    lambda. This name can be used by Reference() instances in the body of the
    lambda to refer to the parameter.
*   <b>`parameter_type`</b>: The type of the parameter, an instance of
    types.Type or something convertible to it by types.to_type().
*   <b>`result`</b>: The resulting value produced by the expression that forms
    the body of the lambda. Must be an instance of ComputationBuildingBlock.

#### Raises:

*   <b>`TypeError`</b>: if the arguments are of the wrong types.

## Properties

<h3 id="parameter_name"><code>parameter_name</code></h3>

<h3 id="parameter_type"><code>parameter_type</code></h3>

<h3 id="proto"><code>proto</code></h3>

Returns a serialized form of this object as a pb.Computation instance.

<h3 id="result"><code>result</code></h3>

<h3 id="type_signature"><code>type_signature</code></h3>

Returns the TFF type of this object (an instance of
<a href="../../tff/Type.md"><code>tff.Type</code></a>).

## Methods

<h3 id="compact_representation"><code>compact_representation</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py">View
source</a>

```python
compact_representation()
```

Returns the compact string representation of this building block.

<h3 id="formatted_representation"><code>formatted_representation</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py">View
source</a>

```python
formatted_representation()
```

Returns the formatted string representation of this building block.

<h3 id="from_proto"><code>from_proto</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py">View
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

<h3 id="structural_representation"><code>structural_representation</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/compiler/building_blocks.py">View
source</a>

```python
structural_representation()
```

Returns the structural string representation of this building block.

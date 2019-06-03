<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.Block" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="locals"/>
<meta itemprop="property" content="proto"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="tff_repr"/>
<meta itemprop="property" content="type_signature"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_proto"/>
</div>

# tff.framework.Block

## Class `Block`

A representation of a block of code in TFF's internal language.

Inherits From:
[`ComputationBuildingBlock`](../../tff/framework/ComputationBuildingBlock.md)

Defined in
[`python/core/impl/computation_building_blocks.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_building_blocks.py).

<!-- Placeholder for "Used in" -->

A block is a syntactic structure that consists of a sequence of local name
bindings followed by a result. The bindings are interpreted sequentially, with
bindings later in the sequence in the scope of those listed earlier, and the
result in the scope of the entire sequence. The usual hiding rules apply.

An informal concise notation for blocks is the following, with `name_k`
representing the names defined locally for the block, `value_k` the values
associated with them, and `result` being the expression that reprsents the value
of the block construct.

```
let name_1=value_1, name_2=value_2, ..., name_n=value_n in result
```

Blocks are technically a redundant abstraction, as they can be equally well
represented by lambda expressions. A block of the form `let x=y in z` is roughly
equivalent to `(x -> z)(y)`. Although redundant, blocks have a use as a way to
reduce TFF computation ASTs to a simpler, less nested and more readable form,
and are helpful in AST transformations as a mechanism that prevents possible
naming conflicts.

An example use of a block expression to flatten a nested structure below:

```
z = federated_sum(federated_map(x, federated_broadcast(y)))
```

An equivalent form in a more sequential notation using a block expression: `let
v1 = federated_broadcast(y), v2 = federated_map(x, v1) in federated_sum(v2)`

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    local_symbols,
    result
)
```

Creates a block of TFF code.

#### Args:

*   <b>`local_symbols`</b>: The list of one or more local declarations, each of
    which is a 2-tuple (name, value), with 'name' being the string name of a
    local symbol being defined, and 'value' being the instance of
    ComputationBuildingBlock, the output of which will be locally bound to that
    name.
*   <b>`result`</b>: An instance of ComputationBuildingBlock that computes the
    result.

#### Raises:

*   <b>`TypeError`</b>: if the arguments are of the wrong types.

## Properties

<h3 id="locals"><code>locals</code></h3>

<h3 id="proto"><code>proto</code></h3>

Returns a serialized form of this object as a pb.Computation instance.

<h3 id="result"><code>result</code></h3>

<h3 id="tff_repr"><code>tff_repr</code></h3>

Returns the representation of the instance using TFF syntax.

<h3 id="type_signature"><code>type_signature</code></h3>

Returns the TFF type of this object (an instance of
<a href="../../tff/Type.md"><code>tff.Type</code></a>).

## Methods

<h3 id="from_proto"><code>from_proto</code></h3>

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

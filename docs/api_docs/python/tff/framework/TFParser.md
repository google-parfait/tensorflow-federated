<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.TFParser" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tff.framework.TFParser

## Class `TFParser`

Callable taking subset of TFF AST constructs to CompiledComputations.

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/transformations.py">View
source</a>

<!-- Placeholder for "Used in" -->

When this function is applied via `transformation_utils.transform_postorder` to
a TFF AST node satisfying its assumptions, the tree under this node will be
reduced to a single instance of
`computation_building_blocks.CompiledComputation` representing the same logic.

Notice that this function is designed to be applied to what is essentially a
subtree of a larger TFF AST; once the processing on a single device has been
aligned at the AST level, and placement separated from the logic of this
processing, we should be left with a function wrapped via `federated_map` or
`federated_apply` to a federated argument. It is this function which we need to
reduce to TensorFlow, and it is to the root node of this function which we are
looking to apply `TFParser`. Because of this, we assume that there is a lambda
expression at the top of the AST we are looking to parse, as well as the rest of
the assumptions below.

1.  All called lambdas have been converted to blocks.
2.  All blocks have been inlined; that is, there are no block/LET constructs
    remaining.
3.  All compiled computations are called.
4.  No compiled computations have been partially called; we believe this should
    be handled correctly today but we haven't reasoned explicitly about this
    possibility.
5.  The only leaf nodes present under `comp` are compiled computations and
    references to the argument of the top-level lambda which we are hoping to
    replace with a compiled computation. Further, every leaf node which is a
    reference has as its parent a `computation_building_blocks.Call`, whose
    associated function is a TF graph. This prevents us from needing to deal
    with arbitrary nesting of references and TF graphs, and significantly
    clarifies the reasoning. This can be accomplished by "decorating" the
    appropriate leaves with called identity TF graphs, the construction of which
    is provided by a utility module.
6.  There is only a single lambda binding any references present in the AST, and
    it is placed at the root of the AST to which we apply `TFParser`.
7.  There are no intrinsics present in the AST.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/transformations.py">View
source</a>

```python
__init__()
```

Populates the parser library with mutually exclusive options.

## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/transformations.py">View
source</a>

```python
__call__(comp)
```

Transforms `comp` by checking all elements of the parser library.

This function is roughly performing intermediate-code generation, taking TFF and
generating TF. Calling this function is essentially checking the stack and
selecting a semantic action based on its contents, and *only one* of these
actions should be selected for a given computation.

Notice that since the parser library contains mutually exclusive options, it is
safe to return early.

#### Args:

*   <b>`comp`</b>: The `computation_building_blocks.ComputationBuildingBlock` to
    check for possibility of reduction according to the parsing library.

#### Returns:

A tuple whose first element is a possibly transformed version of `comp`, and
whose second is a Boolean indicating whether or not `comp` was transformed. This
is in conforming to the conventions of
`transformation_utils.transform_postorder`.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.transform_postorder" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.transform_postorder

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/transformation_utils.py">View
source</a>

Traverses `comp` recursively postorder and replaces its constituents.

```python
tff.framework.transform_postorder(
    comp,
    transform
)
```

<!-- Placeholder for "Used in" -->

For each element of `comp` viewed as an expression tree, the transformation
`transform` is applied first to building blocks it is parameterized by, then the
element itself. The transformation `transform` should act as an identity
function on the kinds of elements (computation building blocks) it does not care
to transform. This corresponds to a post-order traversal of the expression tree,
i.e., parameters are alwaysd transformed left-to-right (in the order in which
they are listed in building block constructors), then the parent is visited and
transformed with the already-visited, and possibly transformed arguments in
place.

NOTE: In particular, in `Call(f,x)`, both `f` and `x` are arguments to `Call`.
Therefore, `f` is transformed into `f'`, next `x` into `x'` and finally,
`Call(f',x')` is transformed at the end.

#### Args:

*   <b>`comp`</b>: A `computation_building_block.ComputationBuildingBlock` to
    traverse and transform bottom-up.
*   <b>`transform`</b>: The transformation to apply locally to each building
    block in `comp`. It is a Python function that accepts a building block at
    input, and should return a (building block, bool) tuple as output, where the
    building block is a `computation_building_block.ComputationBuildingBlock`
    representing either the original building block or a transformed building
    block and the bool is a flag indicating if the building block was modified
    as.

#### Returns:

The result of applying `transform` to parts of `comp` in a bottom-up fashion,
along with a Boolean with the value `True` if `comp` was transformed and `False`
if it was not.

#### Raises:

*   <b>`TypeError`</b>: If the arguments are of the wrong computation_types.
*   <b>`NotImplementedError`</b>: If the argument is a kind of computation
    building block that is currently not recognized.

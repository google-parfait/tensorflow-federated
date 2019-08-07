<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.remove_lambdas_and_blocks" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.remove_lambdas_and_blocks

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/transformations.py">View
source</a>

Removes any called lambdas and blocks from `comp`.

```python
tff.framework.remove_lambdas_and_blocks(comp)
```

<!-- Placeholder for "Used in" -->

This function will rename all the variables in `comp` in a single walk of the
AST, then replace called lambdas with blocks in another walk, since this
transformation interacts with scope in delicate ways. It will chain inlining the
blocks and collapsing the selection-from-tuple pattern together into a final
pass.

#### Args:

*   <b>`comp`</b>: Instance of
    `computation_building_blocks.ComputationBuildingBlock` from which we want to
    remove called lambdas and blocks.

#### Returns:

A transformed version of `comp` which has no called lambdas or blocks, and no
extraneous selections from tuples.

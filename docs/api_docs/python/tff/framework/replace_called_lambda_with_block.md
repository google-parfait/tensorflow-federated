<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.replace_called_lambda_with_block" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.replace_called_lambda_with_block

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/transformations.py">View
source</a>

Replaces all the called lambdas in `comp` with a block.

```python
tff.framework.replace_called_lambda_with_block(comp)
```

<!-- Placeholder for "Used in" -->

This transform traverses `comp` postorder, matches the following pattern, and
replaces the following computation containing a called lambda:

          Call
         /    \

Lambda(x) Comp(y) \
Comp(z)

(x -> z)(y)

with the following computation containing a block:

           Block
          /     \

[x=Comp(y)] Comp(z)

let x=y in z

The functional computation `b` and the argument `c` are retained; the other
computations are replaced. This transformation is used to facilitate the merging
of TFF orchestration logic, in particular to remove unnecessary lambda
expressions and as a stepping stone for merging Blocks together.

#### Args:

*   <b>`comp`</b>: The computation building block in which to perform the
    replacements.

#### Returns:

A new computation with the transformation applied or the original `comp`.

#### Raises:

*   <b>`TypeError`</b>: If types do not match.

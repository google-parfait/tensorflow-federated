<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.insert_called_tf_identity_at_leaves" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.insert_called_tf_identity_at_leaves

Inserts an identity TF graph called on References under `comp`.

```python
tff.framework.insert_called_tf_identity_at_leaves(comp)
```

Defined in
[`python/core/impl/transformations.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/transformations.py).

<!-- Placeholder for "Used in" -->

For ease of reasoning about and proving completeness of TFF-to-TF translation
capabilities, we will maintain the invariant that we constantly pass up the AST
instances of the pattern:

                                  Call
                                /      \
            CompiledComputation         Reference

Any block of TFF reducible to TensorFlow must have a functional type signature
without nested functions, and therefore we may assume there is a single
Reference in the code we are parsing to TF. We continually push logic into the
compiled computation as we make our way up the AST, preserving the pattern
above; when we hit the lambda that binds this reference, we simply unwrap the
call.

To perform this process, we must begin with this pattern; otherwise there may be
some arbitrary TFF constructs present between any occurrences of TF and the
arguments to which they are applied, e.g. arbitrary selections from and nesting
of tuples containing references.

`insert_called_tf_identity_at_leaves` ensures that the pattern above is present
at the leaves of any portion of the TFF AST which is destined to be reduced to
TF.

We detect such a destiny by checking for the existence of a
`computation_building_blocks.Lambda` whose parameter and result type can both be
bound into TensorFlow. This pattern is enforced here as parameter validation on
`comp`.

#### Args:

*   <b>`comp`</b>: Instance of `computation_building_blocks.Lambda` whose AST we
    will traverse, replacing appropriate instances of
    `computation_building_blocks.Reference` with graphs representing thei
    identity function of the appropriate type called on the same reference.
    `comp` must declare a parameter and result type which are both able to be
    stamped in to a TensorFlow graph.

#### Returns:

A possibly modified version of `comp`, where any references now have a parent of
type `computation_building_blocks.Call` with function an instance of
`computation_building_blocks.CompiledComputation`.

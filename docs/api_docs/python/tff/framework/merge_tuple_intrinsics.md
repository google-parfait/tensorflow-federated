<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.merge_tuple_intrinsics" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.merge_tuple_intrinsics

Merges all the tuples of intrinsics in `comp` into one intrinsic.

```python
tff.framework.merge_tuple_intrinsics(
    comp,
    uri
)
```

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/transformations.py">View
source</a>

<!-- Placeholder for "Used in" -->

This transform traverses `comp` postorder, matches the following pattern, and
replaces the following computation containing a tuple of called intrinsics all
represeting the same operation:

         Tuple
         |
         [Call,                        Call, ...]
         /    \                       /    \

Intrinsic Tuple Intrinsic Tuple | | [Comp(f1), Comp(v1), ...] [Comp(f2),
Comp(v2), ...]

<Intrinsic(<f1, v1>), Intrinsic(<f2, v2>)>

with the following computation containing one called intrinsic:

federated_unzip(Call) / \
Intrinsic Tuple | [Block, federated_zip(Tuple), ...] / \ | [fn=Tuple]
Lambda(arg) [Comp(v1), Comp(v2), ...] | \
[Comp(f1), Comp(f2), ...] Tuple | [Call, Call, ...] / \ / \
Sel(0) Sel(0) Sel(1) Sel(1) / / / / Ref(fn) Ref(arg) Ref(fn) Ref(arg)

Intrinsic(< (let fn=<f1, f2> in (arg -> <fn[0](arg[0]), fn[1](arg[1])>)),
<v1, v2>,

> )

The functional computations `f1`, `f2`, etc..., and the computations `v1`, `v2`,
etc... are retained; the other computations are replaced.

NOTE: This is just an example of what this transformation would look like when
applied to a tuple of federated maps. The components `f1`, `f2`, `v1`, and `v2`
and the number of those components are not important.

This transformation is implemented to match the following intrinsics:

*   intrinsic_defs.FEDERATED_AGGREGATE.uri
*   intrinsic_defs.FEDERATED_APPLY.uri
*   intrinsic_defs.FEDERATED_BROADCAST.uri
*   intrinsic_defs.FEDERATED_MAP.uri

#### Args:

*   <b>`comp`</b>: The computation building block in which to perform the
    merges.
*   <b>`uri`</b>: The URI of the intrinsic to merge.

#### Returns:

A new computation with the transformation applied or the original `comp`.

#### Raises:

*   <b>`TypeError`</b>: If types do not match.

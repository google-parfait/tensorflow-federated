<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="FEDERATED_AGGREGATE"/>
<meta itemprop="property" content="FEDERATED_BROADCAST"/>
<meta itemprop="property" content="FEDERATED_MAP"/>
<meta itemprop="property" content="FEDERATED_MAP_ALL_EQUAL"/>
</div>

# Module: tff.framework

Interfaces for extensions, selectively lifted out of `impl`.

Defined in
[`python/core/framework/__init__.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/framework/__init__.py).

<!-- Placeholder for "Used in" -->

## Classes

[`class Block`](../tff/framework/Block.md): A representation of a block of code
in TFF's internal language.

[`class Call`](../tff/framework/Call.md): A representation of a function
invocation in TFF's internal language.

[`class CompiledComputation`](../tff/framework/CompiledComputation.md): A
representation of a fully constructed and serialized computation.

[`class ComputationBuildingBlock`](../tff/framework/ComputationBuildingBlock.md):
The abstract base class for abstractions in the TFF's internal language.

[`class Intrinsic`](../tff/framework/Intrinsic.md): A representation of an
intrinsic in TFF's internal language.

[`class Lambda`](../tff/framework/Lambda.md): A representation of a lambda
expression in TFF's internal language.

[`class Placement`](../tff/framework/Placement.md): A representation of a
placement literal in TFF's internal language.

[`class Reference`](../tff/framework/Reference.md): A reference to a name
defined earlier in TFF's internal language.

[`class Selection`](../tff/framework/Selection.md): A selection by name or index
from a tuple-typed value in TFF's language.

[`class Tuple`](../tff/framework/Tuple.md): A tuple with named or unnamed
elements in TFF's internal language.

## Functions

[`check_has_unique_names(...)`](../tff/framework/check_has_unique_names.md)

[`create_federated_map_all_equal(...)`](../tff/framework/create_federated_map_all_equal.md):
Creates a called federated map of equal values.

[`create_federated_map_or_apply(...)`](../tff/framework/create_federated_map_or_apply.md):
Creates a called federated map or apply depending on `arg`s placement.

[`create_federated_zip(...)`](../tff/framework/create_federated_zip.md): Creates
a called federated zip.

[`get_map_of_unbound_references(...)`](../tff/framework/get_map_of_unbound_references.md):
Gets a Python `dict` of the unbound references in `comp`.

[`inline_block_locals(...)`](../tff/framework/inline_block_locals.md): Inlines
the block variables in `comp` whitelisted by `variable_names`.

[`is_assignable_from(...)`](../tff/framework/is_assignable_from.md): Determines
whether `target_type` is assignable from `source_type`.

[`is_called_intrinsic(...)`](../tff/framework/is_called_intrinsic.md): Returns
`True` if `comp` is a called intrinsic with the `uri` or `uri`s.

[`merge_tuple_intrinsics(...)`](../tff/framework/merge_tuple_intrinsics.md):
Merges all the tuples of intrinsics in `comp` into one intrinsic.

[`remove_mapped_or_applied_identity(...)`](../tff/framework/remove_mapped_or_applied_identity.md):
Removes all the mapped or applied identity functions in `comp`.

[`replace_called_lambda_with_block(...)`](../tff/framework/replace_called_lambda_with_block.md):
Replaces all the called lambdas in `comp` with a block.

[`replace_selection_from_tuple_with_element(...)`](../tff/framework/replace_selection_from_tuple_with_element.md):
Replaces any selection from a tuple with the underlying tuple element.

[`transform_postorder(...)`](../tff/framework/transform_postorder.md): Traverses
`comp` recursively postorder and replaces its constituents.

[`type_from_tensors(...)`](../tff/framework/type_from_tensors.md): Builds a
<a href="../tff/Type.md"><code>tff.Type</code></a> from supplied tensors.

[`type_to_tf_tensor_specs(...)`](../tff/framework/type_to_tf_tensor_specs.md):
Returns nested structure of `tf.TensorSpec`s for a given TFF type.

[`unique_name_generator(...)`](../tff/framework/unique_name_generator.md):
Yields a new unique name that does not exist in `comp`.

[`uniquify_reference_names(...)`](../tff/framework/uniquify_reference_names.md):
Replaces all the reference names in `comp` with unique names.

## Other Members

*   `FEDERATED_AGGREGATE` <a id="FEDERATED_AGGREGATE"></a>
*   `FEDERATED_BROADCAST` <a id="FEDERATED_BROADCAST"></a>
*   `FEDERATED_MAP` <a id="FEDERATED_MAP"></a>
*   `FEDERATED_MAP_ALL_EQUAL` <a id="FEDERATED_MAP_ALL_EQUAL"></a>

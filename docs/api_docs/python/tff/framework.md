<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="FEDERATED_AGGREGATE"/>
<meta itemprop="property" content="FEDERATED_APPLY"/>
<meta itemprop="property" content="FEDERATED_BROADCAST"/>
<meta itemprop="property" content="FEDERATED_MAP"/>
<meta itemprop="property" content="FEDERATED_MAP_ALL_EQUAL"/>
</div>

# Module: tff.framework

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/framework/__init__.py">View
source</a>

Interfaces for extensions, selectively lifted out of `impl`.

<!-- Placeholder for "Used in" -->

## Classes

[`class Block`](../tff/framework/Block.md): A representation of a block of code
in TFF's internal language.

[`class CachingExecutor`](../tff/framework/CachingExecutor.md): The caching
executor only performs caching.

[`class Call`](../tff/framework/Call.md): A representation of a function
invocation in TFF's internal language.

[`class CompiledComputation`](../tff/framework/CompiledComputation.md): A
representation of a fully constructed and serialized computation.

[`class ComputationBuildingBlock`](../tff/framework/ComputationBuildingBlock.md):
The abstract base class for abstractions in the TFF's internal language.

[`class ConcurrentExecutor`](../tff/framework/ConcurrentExecutor.md): The
concurrent executor delegates work to a separate thread.

[`class EagerExecutor`](../tff/framework/EagerExecutor.md): The eager executor
only runs TensorFlow, synchronously, in eager mode.

[`class Executor`](../tff/framework/Executor.md): Represents the abstract
interface that all executors must implement.

[`class ExecutorService`](../tff/framework/ExecutorService.md): A wrapper around
a target executor that makes it into a gRPC service.

[`class ExecutorValue`](../tff/framework/ExecutorValue.md): Represents the
abstract interface for values embedded within executors.

[`class FederatedExecutor`](../tff/framework/FederatedExecutor.md): The
federated executor orchestrates federated computations.

[`class Intrinsic`](../tff/framework/Intrinsic.md): A representation of an
intrinsic in TFF's internal language.

[`class Lambda`](../tff/framework/Lambda.md): A representation of a lambda
expression in TFF's internal language.

[`class LambdaExecutor`](../tff/framework/LambdaExecutor.md): The lambda
executor handles lambda expressions and related abstractions.

[`class Placement`](../tff/framework/Placement.md): A representation of a
placement literal in TFF's internal language.

[`class Reference`](../tff/framework/Reference.md): A reference to a name
defined earlier in TFF's internal language.

[`class RemoteExecutor`](../tff/framework/RemoteExecutor.md): The remote
executor is a local proxy for a remote executor instance.

[`class Selection`](../tff/framework/Selection.md): A selection by name or index
from a tuple-typed value in TFF's language.

[`class TFParser`](../tff/framework/TFParser.md): Callable taking subset of TFF
AST constructs to CompiledComputations.

[`class TransformingExecutor`](../tff/framework/TransformingExecutor.md): This
executor transforms computations prior to executing them.

[`class Tuple`](../tff/framework/Tuple.md): A tuple with named or unnamed
elements in TFF's internal language.

## Functions

[`are_equivalent_types(...)`](../tff/framework/are_equivalent_types.md):
Determines whether `type1` and `type2` are equivalent.

[`building_block_to_computation(...)`](../tff/framework/building_block_to_computation.md):
Converts a computation building block to a computation impl.

[`check_has_unique_names(...)`](../tff/framework/check_has_unique_names.md)

[`check_intrinsics_whitelisted_for_reduction(...)`](../tff/framework/check_intrinsics_whitelisted_for_reduction.md):
Checks whitelist of intrinsics reducible to aggregate or broadcast.

[`create_federated_map_all_equal(...)`](../tff/framework/create_federated_map_all_equal.md):
Creates a called federated map of equal values.

[`create_federated_map_or_apply(...)`](../tff/framework/create_federated_map_or_apply.md):
Creates a called federated map or apply depending on `arg`s placement.

[`create_federated_zip(...)`](../tff/framework/create_federated_zip.md): Creates
a called federated zip.

[`create_local_executor(...)`](../tff/framework/create_local_executor.md):
Constructs an executor to execute computations on the local machine.

[`get_map_of_unbound_references(...)`](../tff/framework/get_map_of_unbound_references.md):
Gets a Python `dict` of the unbound references in `comp`.

[`inline_block_locals(...)`](../tff/framework/inline_block_locals.md): Inlines
the block variables in `comp` whitelisted by `variable_names`.

[`insert_called_tf_identity_at_leaves(...)`](../tff/framework/insert_called_tf_identity_at_leaves.md):
Inserts an identity TF graph called on References under `comp`.

[`is_assignable_from(...)`](../tff/framework/is_assignable_from.md): Determines
whether `target_type` is assignable from `source_type`.

[`is_called_intrinsic(...)`](../tff/framework/is_called_intrinsic.md): Tests if
`comp` is a called intrinsic with the given `uri`.

[`is_tensorflow_compatible_type(...)`](../tff/framework/is_tensorflow_compatible_type.md):
Checks `type_spec` against an explicit whitelist for `tf_computation`.

[`merge_tuple_intrinsics(...)`](../tff/framework/merge_tuple_intrinsics.md):
Merges all the tuples of intrinsics in `comp` into one intrinsic.

[`remove_lambdas_and_blocks(...)`](../tff/framework/remove_lambdas_and_blocks.md):
Removes any called lambdas and blocks from `comp`.

[`remove_mapped_or_applied_identity(...)`](../tff/framework/remove_mapped_or_applied_identity.md):
Removes all the mapped or applied identity functions in `comp`.

[`replace_called_lambda_with_block(...)`](../tff/framework/replace_called_lambda_with_block.md):
Replaces all the called lambdas in `comp` with a block.

[`replace_intrinsics_with_bodies(...)`](../tff/framework/replace_intrinsics_with_bodies.md):
Reduces intrinsics to their bodies as defined in `intrinsic_bodies.py`.

[`set_default_executor(...)`](../tff/framework/set_default_executor.md): Places
an `executor`-backed execution context at the top of the stack.

[`transform_postorder(...)`](../tff/framework/transform_postorder.md): Traverses
`comp` recursively postorder and replaces its constituents.

[`transform_type_postorder(...)`](../tff/framework/transform_type_postorder.md):
Walks type tree of `type_signature` postorder, calling `transform_fn`.

[`type_from_tensors(...)`](../tff/framework/type_from_tensors.md): Builds a
<a href="../tff/Type.md"><code>tff.Type</code></a> from supplied tensors.

[`type_to_tf_tensor_specs(...)`](../tff/framework/type_to_tf_tensor_specs.md):
Returns nested structure of `tf.TensorSpec`s for a given TFF type.

[`unique_name_generator(...)`](../tff/framework/unique_name_generator.md):
Yields a new unique name that does not exist in `comp`.

[`uniquify_reference_names(...)`](../tff/framework/uniquify_reference_names.md):
Replaces all the bound reference names in `comp` with unique names.

[`unwrap_placement(...)`](../tff/framework/unwrap_placement.md): Strips `comp`'s
placement, returning a single call to map, apply or value.

## Other Members

*   `FEDERATED_AGGREGATE` <a id="FEDERATED_AGGREGATE"></a>
*   `FEDERATED_APPLY` <a id="FEDERATED_APPLY"></a>
*   `FEDERATED_BROADCAST` <a id="FEDERATED_BROADCAST"></a>
*   `FEDERATED_MAP` <a id="FEDERATED_MAP"></a>
*   `FEDERATED_MAP_ALL_EQUAL` <a id="FEDERATED_MAP_ALL_EQUAL"></a>

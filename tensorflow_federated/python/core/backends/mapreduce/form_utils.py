# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for converting to/from the MapReduce form.

Note: Refer to `get_computation_for_map_reduce_form()` for the meaning of
variable names used in this module.
"""

from collections.abc import Callable
from typing import Optional

import federated_language
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.backends.mapreduce import compiler
from tensorflow_federated.python.core.backends.mapreduce import forms
from tensorflow_federated.python.core.backends.mapreduce import intrinsics as mapreduce_intrinsics
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_building_block_factory
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_tree_transformations
from tensorflow_federated.python.core.impl.compiler import transformations
from tensorflow_federated.python.core.impl.compiler import tree_transformations

_GRAPPLER_DEFAULT_CONFIG = tf.compat.v1.ConfigProto()
_AGGRESSIVE = _GRAPPLER_DEFAULT_CONFIG.graph_options.rewrite_options.AGGRESSIVE
_GRAPPLER_DEFAULT_CONFIG.graph_options.rewrite_options.memory_optimization = (
    _AGGRESSIVE
)
_GRAPPLER_DEFAULT_CONFIG.graph_options.rewrite_options.constant_folding = (
    _AGGRESSIVE
)
_GRAPPLER_DEFAULT_CONFIG.graph_options.rewrite_options.arithmetic_optimization = (
    _AGGRESSIVE
)
_GRAPPLER_DEFAULT_CONFIG.graph_options.rewrite_options.loop_optimization = (
    _AGGRESSIVE
)
_GRAPPLER_DEFAULT_CONFIG.graph_options.rewrite_options.function_optimization = (
    _AGGRESSIVE
)

BuildingBlockFn = Callable[
    [federated_language.framework.ComputationBuildingBlock],
    federated_language.framework.ComputationBuildingBlock,
]


def get_computation_for_broadcast_form(
    bf: forms.BroadcastForm,
) -> federated_language.framework.Computation:
  """Creates `federated_language.Computation` from a broadcast form."""
  py_typecheck.check_type(bf, forms.BroadcastForm)
  server_data_type = bf.compute_server_context.type_signature.parameter
  client_data_type = bf.client_processing.type_signature.parameter[1]
  comp_parameter_type = federated_language.StructType([
      (
          bf.server_data_label,
          federated_language.FederatedType(
              server_data_type, federated_language.SERVER
          ),
      ),
      (
          bf.client_data_label,
          federated_language.FederatedType(
              client_data_type, federated_language.CLIENTS
          ),
      ),
  ])

  @federated_language.federated_computation(comp_parameter_type)
  def computation(arg):
    server_data, client_data = arg
    context_at_server = federated_language.federated_map(
        bf.compute_server_context, server_data
    )
    context_at_clients = federated_language.federated_broadcast(
        context_at_server
    )
    client_processing_arg = federated_language.federated_zip(
        (context_at_clients, client_data)
    )
    return federated_language.federated_map(
        bf.client_processing, client_processing_arg
    )

  return computation


def get_state_initialization_computation(
    initialize_computation: federated_language.framework.ConcreteComputation,
    grappler_config: tf.compat.v1.ConfigProto = _GRAPPLER_DEFAULT_CONFIG,
) -> federated_language.framework.Computation:
  """Validates and transforms a computation to generate state.

  Args:
    initialize_computation: A `federated_language.framework.ConcreteComputation`
      that should generate initial state for a computation that is compatible
      with a federated learning system that implements the contract of a backend
      defined in the backends/mapreduce directory.
    grappler_config: An optional instance of `tf.compat.v1.ConfigProto` to
      configure Grappler graph optimization of the TensorFlow graphs. These
      options are combined with a set of defaults that aggressively configure
      Grappler. If the input `grappler_config` has
      `graph_options.rewrite_options.disable_meta_optimizer=True`, Grappler is
      bypassed.

  Returns:
    A `federated_language.framework.Computation` that can generate state for a
    computation.

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  init_type = initialize_computation.type_signature
  _check_type_is_no_arg_fn(init_type, '`initialize`', TypeError)
  if (
      not isinstance(init_type.result, federated_language.FederatedType)
      or init_type.result.placement is not federated_language.SERVER
  ):
    raise TypeError(
        'Expected `initialize` to return a single federated value '
        'placed at server (type `T@SERVER`), found return type:\n'
        f'{init_type.result}'  # pytype: disable=attribute-error
    )
  initialize_tree = initialize_computation.to_building_block()
  initialize_tree, _ = (
      tensorflow_tree_transformations.replace_intrinsics_with_bodies(
          initialize_tree
      )
  )
  federated_language.framework.check_contains_only_reducible_intrinsics(
      initialize_tree
  )
  initialize_tree = compiler.consolidate_and_extract_local_processing(
      initialize_tree, grappler_config
  )
  return federated_language.framework.ConcreteComputation(
      computation_proto=initialize_tree.to_proto(),
      context_stack=federated_language.framework.get_context_stack(),
  )


def get_computation_for_map_reduce_form(
    mrf: forms.MapReduceForm,
) -> federated_language.framework.Computation:
  """Creates `federated_language.Computation` from a MapReduce form.

  Args:
    mrf: An instance of `tff.backends.mapreduce.MapReduceForm`.

  Returns:
    An instance of `federated_language.Computation` that corresponds to `mrf`.

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  py_typecheck.check_type(mrf, forms.MapReduceForm)

  @federated_language.federated_computation(mrf.type_signature.parameter)
  def computation(arg):
    """The logic of a single MapReduce processing round."""
    server_state, client_data = arg
    broadcast_input = federated_language.federated_map(
        mrf.prepare, server_state
    )
    broadcast_result = federated_language.federated_broadcast(broadcast_input)
    work_arg = federated_language.federated_zip([client_data, broadcast_result])
    (
        aggregate_input,
        secure_sum_bitwidth_input,
        secure_sum_input,
        secure_modular_sum_input,
    ) = federated_language.federated_map(mrf.work, work_arg)
    aggregate_result = federated_language.federated_aggregate(
        aggregate_input, mrf.zero(), mrf.accumulate, mrf.merge, mrf.report
    )
    secure_sum_bitwidth_result = (
        federated_language.federated_secure_sum_bitwidth(
            secure_sum_bitwidth_input, mrf.secure_sum_bitwidth()
        )
    )
    secure_sum_result = federated_language.federated_secure_sum(
        secure_sum_input, mrf.secure_sum_max_input()
    )
    secure_modular_sum_result = (
        mapreduce_intrinsics.federated_secure_modular_sum(
            secure_modular_sum_input, mrf.secure_modular_sum_modulus()
        )
    )
    update_arg = federated_language.federated_zip((
        server_state,
        (
            aggregate_result,
            secure_sum_bitwidth_result,
            secure_sum_result,
            secure_modular_sum_result,
        ),
    ))
    updated_server_state, server_output = federated_language.federated_map(
        mrf.update, update_arg
    )
    return updated_server_state, server_output

  return computation


def get_computation_for_distribute_aggregate_form(
    daf: forms.DistributeAggregateForm,
) -> federated_language.framework.Computation:
  """Creates `federated_language.Computation` from a DistributeAggregate form.

  Args:
    daf: An instance of `tff.backends.mapreduce.DistributeAggregateForm`.

  Returns:
    An instance of `federated_language.Computation` that corresponds to `daf`.

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  py_typecheck.check_type(daf, forms.DistributeAggregateForm)

  @federated_language.federated_computation(daf.type_signature.parameter)
  def computation(arg):
    """The logic of a single federated computation round."""
    server_state, client_data = arg
    broadcast_input, temp_server_state = daf.server_prepare(server_state)
    broadcast_output = daf.server_to_client_broadcast(broadcast_input)
    aggregation_input = daf.client_work(client_data, broadcast_output)
    aggregation_output = daf.client_to_server_aggregation(
        temp_server_state, aggregation_input
    )
    updated_server_state, server_output = daf.server_result(
        temp_server_state, aggregation_output
    )
    return updated_server_state, server_output

  return computation


def _check_type_is_fn(
    target: federated_language.Type,
    name: str,
    err_fn: Callable[[str], Exception] = compiler.MapReduceFormCompilationError,
):
  if not isinstance(target, federated_language.FunctionType):
    raise err_fn(
        f'Expected {name} to be a function, but {name} had type {target}.'
    )


def _check_type_is_no_arg_fn(
    target: federated_language.Type,
    name: str,
    err_fn: Callable[[str], Exception] = compiler.MapReduceFormCompilationError,
):
  _check_type_is_fn(target, name, err_fn)
  if target.parameter is not None:  # pytype: disable=attribute-error
    raise err_fn(
        f'Expected {name} to take no argument, but found '
        f'parameter of type {target.parameter}.'  # pytype: disable=attribute-error
    )


def _check_function_signature_compatible_with_broadcast_form(
    function_type: federated_language.FunctionType,
):
  """Tests compatibility with `tff.backends.mapreduce.BroadcastForm`."""
  py_typecheck.check_type(function_type, federated_language.FunctionType)
  if not (
      isinstance(function_type.parameter, federated_language.StructType)
      and len(function_type.parameter) == 2
  ):
    raise TypeError(
        '`BroadcastForm` requires a computation which accepts two arguments '
        '(server data and client data) but found parameter type:\n'
        f'{function_type.parameter}'
    )
  server_data_type, client_data_type = function_type.parameter  # pytype: disable=attribute-error
  if (
      not isinstance(server_data_type, federated_language.FederatedType)
      or server_data_type.placement is not federated_language.SERVER
  ):
    raise TypeError(
        '`BroadcastForm` expects a computation whose first parameter is server '
        'data (a federated type placed at server) but found first parameter of '
        f'type:\n{server_data_type}'
    )
  if (
      not isinstance(client_data_type, federated_language.FederatedType)
      or client_data_type.placement is not federated_language.CLIENTS
  ):
    raise TypeError(
        '`BroadcastForm` expects a computation whose first parameter is client '
        'data (a federated type placed at clients) but found first parameter '
        f'of type:\n{client_data_type}'
    )
  result_type = function_type.result
  if (
      not isinstance(result_type, federated_language.FederatedType)
      or result_type.placement is not federated_language.CLIENTS
  ):
    raise TypeError(
        '`BroadcastForm` expects a computation whose result is client data '
        '(a federated type placed at clients) but found result type:\n'
        f'{result_type}'
    )


def _check_contains_only_reducible_intrinsics(
    comp: federated_language.framework.ComputationBuildingBlock,
):
  """Checks that `comp` contains intrinsics reducible to aggregate or broadcast.

  Args:
    comp: Instance of `federated_language.framework.ComputationBuildingBlock` to
      check for presence of intrinsics not currently immediately reducible to
      `FEDERATED_AGGREGATE` or `FEDERATED_BROADCAST`, or local processing.

  Raises:
    ValueError: If we encounter an intrinsic under `comp` that is not reducible.
  """
  reducible_uris = (
      federated_language.framework.FEDERATED_AGGREGATE.uri,
      federated_language.framework.FEDERATED_APPLY.uri,
      federated_language.framework.FEDERATED_BROADCAST.uri,
      federated_language.framework.FEDERATED_EVAL_AT_CLIENTS.uri,
      federated_language.framework.FEDERATED_EVAL_AT_SERVER.uri,
      federated_language.framework.FEDERATED_MAP_ALL_EQUAL.uri,
      federated_language.framework.FEDERATED_MAP.uri,
      federated_language.framework.FEDERATED_SECURE_SUM_BITWIDTH.uri,
      federated_language.framework.FEDERATED_SECURE_SUM.uri,
      federated_language.framework.FEDERATED_VALUE_AT_CLIENTS.uri,
      federated_language.framework.FEDERATED_VALUE_AT_SERVER.uri,
      federated_language.framework.FEDERATED_ZIP_AT_CLIENTS.uri,
      federated_language.framework.FEDERATED_ZIP_AT_SERVER.uri,
      mapreduce_intrinsics.FEDERATED_SECURE_MODULAR_SUM.uri,
  )

  def _check(comp):
    if (
        isinstance(comp, federated_language.framework.Intrinsic)
        and comp.uri not in reducible_uris
    ):
      raise ValueError(
          'Encountered an Intrinsic not currently reducible to aggregate or '
          'broadcast, the intrinsic {}'.format(comp.compact_representation())
      )

  federated_language.framework.visit_postorder(comp, _check)


def check_computation_compatible_with_map_reduce_form(
    comp: federated_language.framework.ConcreteComputation,
    *,
    tff_internal_preprocessing: Optional[BuildingBlockFn] = None,
) -> federated_language.framework.ComputationBuildingBlock:
  """Tests compatibility with `tff.backends.mapreduce.MapReduceForm`.

  Note: the conditions here are specified in the documentation for
    `get_map_reduce_form_for_computation`. Changes to this function should
    be propagated to that documentation.

  Args:
    comp: An instance of `federated_language.framework.ConcreteComputation` to
      check for compatibility with `tff.backends.mapreduce.MapReduceForm`.
    tff_internal_preprocessing: An optional function to transform the AST of the
      computation.

  Returns:
    A TFF-internal building-block representing the validated and simplified
    computation.

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  py_typecheck.check_type(
      comp, federated_language.framework.ConcreteComputation
  )
  comp_tree = comp.to_building_block()
  if tff_internal_preprocessing is not None:
    comp_tree = tff_internal_preprocessing(comp_tree)

  comp_type = comp_tree.type_signature
  _check_type_is_fn(comp_type, '`comp`', TypeError)
  if (
      not isinstance(comp_type.parameter, federated_language.StructType)
      or len(comp_type.parameter) != 2
  ):  # pytype: disable=attribute-error
    raise TypeError(
        'Expected `comp` to take two arguments, found parameter '
        f' type:\n{comp_type.parameter}'  # pytype: disable=attribute-error
    )
  if (
      not isinstance(comp_type.result, federated_language.StructType)
      or len(comp_type.result) != 2
  ):  # pytype: disable=attribute-error
    raise TypeError(
        'Expected `comp` to return two values, found result '
        f'type:\n{comp_type.result}'  # pytype: disable=attribute-error
    )

  comp_tree, _ = tensorflow_tree_transformations.replace_intrinsics_with_bodies(
      comp_tree
  )
  comp_tree = _replace_lambda_body_with_call_dominant_form(comp_tree)

  _check_contains_only_reducible_intrinsics(comp_tree)
  federated_language.framework.check_broadcast_not_dependent_on_aggregate(
      comp_tree
  )

  return comp_tree


def _untuple_broadcast_only_before_after(before, after):
  """Removes the tuple-ing of the `broadcast` params and results."""
  # Since there is only a single intrinsic here, there's no need for the outer
  # `{intrinsic_name}_param`/`{intrinsic_name}_result` tuples.
  untupled_before = federated_language.framework.select_output_from_lambda(
      before, 'federated_broadcast_param'
  )
  after_param_name = next(
      federated_language.framework.unique_name_generator(after)
  )
  after_param_type = federated_language.StructType([
      ('original_arg', after.parameter_type.original_arg),  # pytype: disable=attribute-error
      (
          'federated_broadcast_result',
          after.parameter_type.intrinsic_results.federated_broadcast_result,  # pytype: disable=attribute-error
      ),
  ])
  after_param_ref = federated_language.framework.Reference(
      after_param_name, after_param_type
  )

  after_result_arg = federated_language.framework.Struct([
      (
          'original_arg',
          federated_language.framework.Selection(
              after_param_ref, 'original_arg'
          ),
      ),
      (
          'intrinsic_results',
          federated_language.framework.Struct([(
              'federated_broadcast_result',
              federated_language.framework.Selection(
                  after_param_ref, 'federated_broadcast_result'
              ),
          )]),
      ),
  ])
  after_result = federated_language.framework.Call(
      after,
      after_result_arg,
  )
  untupled_after = federated_language.framework.Lambda(
      after_param_name,
      after_param_type,
      after_result,
  )
  return untupled_before, untupled_after


def _split_ast_on_broadcast(bb):
  """Splits an AST on the `broadcast` intrinsic.

  Args:
    bb: An AST of arbitrary shape, potentially containing a broadcast.

  Returns:
    Two ASTs, the first of which maps comp's input to the
    argument of broadcast, and the second of which maps comp's input and
    broadcast's output to comp's output.
  """
  before, after = transformations.force_align_and_split_by_intrinsics(
      bb, [tensorflow_building_block_factory.create_null_federated_broadcast()]
  )
  return _untuple_broadcast_only_before_after(before, after)


def _split_ast_on_aggregate(bb):
  """Splits an AST on reduced aggregation intrinsics.

  Args:
    bb: An AST to split on `federated_aggregate`,
      `federated_secure_sum_bitwidth`, `federated_secure_sum`, and
      `federated_secure_modular_sum`.

  Returns:
    Two ASTs, the first of which maps comp's input to the arguments
    to `federated_aggregate` and `federated_secure_sum_bitwidth`, and the
    second of which maps comp's input and the output of `federated_aggregate`
    and `federated_secure_sum_bitwidth` to comp's output.
  """
  return transformations.force_align_and_split_by_intrinsics(
      bb,
      [
          tensorflow_building_block_factory.create_null_federated_aggregate(),
          tensorflow_building_block_factory.create_null_federated_secure_sum_bitwidth(),
          tensorflow_building_block_factory.create_null_federated_secure_sum(),
          mapreduce_intrinsics.create_null_federated_secure_modular_sum(),
      ],
  )


def _prepare_for_rebinding(bb):
  """Replaces `bb` with semantically equivalent version for rebinding."""
  bb = tree_transformations.normalize_types(bb)
  bb, _ = tree_transformations.remove_mapped_or_applied_identity(bb)
  bb = transformations.to_call_dominant(bb)
  bb, _ = tree_transformations.remove_unused_block_locals(bb)
  return bb


def _construct_selection_from_federated_tuple(
    federated_tuple: federated_language.framework.ComputationBuildingBlock,
    index: int,
    name_generator,
) -> federated_language.framework.ComputationBuildingBlock:
  """Selects the index `selected_index` from `federated_tuple`."""
  if not isinstance(
      federated_tuple.type_signature, federated_language.FederatedType
  ):
    raise ValueError(
        'Expected a `federated_language.FederatedType`, found'
        f' {federated_tuple.type_signature}.'
    )
  member_type = federated_tuple.type_signature.member
  if not isinstance(member_type, federated_language.StructType):
    raise ValueError(
        f'Expected a `federated_language.StructType`, found {member_type}.'
    )
  param_name = next(name_generator)
  selecting_function = federated_language.framework.Lambda(
      param_name,
      member_type,
      federated_language.framework.Selection(
          federated_language.framework.Reference(param_name, member_type),
          index=index,
      ),
  )
  return federated_language.framework.create_federated_map_or_apply(
      selecting_function, federated_tuple
  )


def _as_function_of_single_subparameter(
    bb: federated_language.framework.Lambda, index: int
) -> federated_language.framework.Lambda:
  """Turns `x -> ...only uses x_i...` into `x_i -> ...only uses x_i`."""
  federated_language.framework.check_has_unique_names(bb)
  bb = _prepare_for_rebinding(bb)
  new_name = next(federated_language.framework.unique_name_generator(bb))
  new_ref = federated_language.framework.Reference(
      new_name, bb.type_signature.parameter[index]
  )
  new_lambda_body = tree_transformations.replace_selections(
      bb.result, bb.parameter_name, {(index,): new_ref}
  )
  new_lambda = federated_language.framework.Lambda(
      new_ref.name, new_ref.type_signature, new_lambda_body
  )
  federated_language.framework.check_contains_no_new_unbound_references(
      bb, new_lambda
  )
  return new_lambda


class _NonFederatedSelectionError(TypeError):
  pass


class _MismatchedSelectionPlacementError(TypeError):
  pass


def _as_function_of_some_federated_subparameters(
    bb: federated_language.framework.Lambda,
    paths,
) -> federated_language.framework.Lambda:
  """Turns `x -> ...only uses parts of x...` into `parts_of_x -> ...`."""
  federated_language.framework.check_has_unique_names(bb)
  bb = _prepare_for_rebinding(bb)
  name_generator = federated_language.framework.unique_name_generator(bb)

  type_list = []
  int_paths = []
  for path in paths:
    selected_type = bb.parameter_type
    int_path = []
    for index in path:
      if not isinstance(selected_type, federated_language.StructType):
        raise tree_transformations.ParameterSelectionError(path, bb)
      if isinstance(index, int):
        if index >= len(selected_type):
          raise tree_transformations.ParameterSelectionError(path, bb)
        int_path.append(index)
      else:
        py_typecheck.check_type(index, str)
        if index not in selected_type.fields():
          raise tree_transformations.ParameterSelectionError(path, bb)
        names = [n for n, _ in selected_type.items()]
        int_path.append(names.index(index))
      selected_type = selected_type[index]
    if not isinstance(selected_type, federated_language.FederatedType):
      raise _NonFederatedSelectionError(
          'Attempted to rebind references to parameter selection path '
          f'{path} from type {bb.parameter_type}, but the value at that path '
          f'was of non-federated type {selected_type}. Selections must all '
          f'be of federated type. Original AST:\n{bb}'
      )
    int_paths.append(tuple(int_path))
    type_list.append(selected_type)

  placement = type_list[0].placement
  if not all(x.placement is placement for x in type_list):
    raise _MismatchedSelectionPlacementError(
        'In order to zip the argument to the lower-level lambda together, all '
        'selected arguments should be at the same placement. Your selections '
        f'have resulted in the list of types:\n{type_list}'
    )

  zip_type = federated_language.FederatedType(
      [x.member for x in type_list], placement=placement
  )
  ref_to_zip = federated_language.framework.Reference(
      next(name_generator), zip_type
  )
  path_to_replacement = {}
  for i, path in enumerate(int_paths):
    path_to_replacement[path] = _construct_selection_from_federated_tuple(
        ref_to_zip, i, name_generator
    )

  new_lambda_body = tree_transformations.replace_selections(
      bb.result, bb.parameter_name, path_to_replacement
  )
  lambda_with_zipped_param = federated_language.framework.Lambda(
      ref_to_zip.name, ref_to_zip.type_signature, new_lambda_body
  )
  federated_language.framework.check_contains_no_new_unbound_references(
      bb, lambda_with_zipped_param
  )

  return lambda_with_zipped_param


def _extract_compute_server_context(before_broadcast, grappler_config):
  """Extracts `compute_server_config` from `before_broadcast`."""
  server_data_index_in_before_broadcast = 0
  compute_server_context = _as_function_of_single_subparameter(
      before_broadcast, server_data_index_in_before_broadcast
  )
  return compiler.consolidate_and_extract_local_processing(
      compute_server_context, grappler_config
  )


def _extract_client_processing(after_broadcast, grappler_config):
  """Extracts `client_processing` from `after_broadcast`."""
  context_from_server_index_in_after_broadcast = (1,)
  client_data_index_in_after_broadcast = (0, 1)
  # NOTE: the order of parameters here is different from `work`.
  # `work` is odd in that it takes its parameters as `(data, params)` rather
  # than `(params, data)` (the order of the iterative process / computation).
  # Here, we use the same `(params, data)` ordering as in the input computation.
  client_processing = _as_function_of_some_federated_subparameters(
      after_broadcast,
      [
          context_from_server_index_in_after_broadcast,
          client_data_index_in_after_broadcast,
      ],
  )
  return compiler.consolidate_and_extract_local_processing(
      client_processing, grappler_config
  )


def _extract_prepare(before_broadcast, grappler_config):
  """extracts `prepare` from `before_broadcast`.

  This function is intended to be used by `get_map_reduce_form_for_computation`
  only. As a result, this function does not assert that `before_broadcast` has
  the expected structure, the caller is expected to perform these checks before
  calling this function.

  Args:
    before_broadcast: The first result of splitting `next_bb` on
      `federated_language.framework.FEDERATED_BROADCAST`.
    grappler_config: An instance of `tf.compat.v1.ConfigProto` to configure
      Grappler graph optimization.

  Returns:
    `prepare` as specified by `forms.MapReduceForm`, an instance of
    `federated_language.framework.CompiledComputation`.

  Raises:
    compiler.MapReduceFormCompilationError: If we extract an AST of the wrong
      type.
  """
  server_state_index_in_before_broadcast = 0
  prepare = _as_function_of_single_subparameter(
      before_broadcast, server_state_index_in_before_broadcast
  )
  return compiler.consolidate_and_extract_local_processing(
      prepare, grappler_config
  )


def _extract_work(before_aggregate, grappler_config):
  """Extracts `work` from `before_aggregate`.

  This function is intended to be used by
  `get_map_reduce_form_for_computation` only. As a result, this function does
  not assert that `before_aggregate` has the expected structure, the caller
  is expected to perform these checks before calling this function.

  Args:
    before_aggregate: The first result of splitting `after_broadcast` on
      aggregate intrinsics.
    grappler_config: An instance of `tf.compat.v1.ConfigProto` to configure
      Grappler graph optimization.

  Returns:
    `work` as specified by `forms.MapReduceForm`, an instance of
    `federated_language.framework.CompiledComputation`.

  Raises:
    compiler.MapReduceFormCompilationError: If we extract an AST of the wrong
      type.
  """
  # Indices of `work` args in `before_aggregate` parameter
  client_data_index = ('original_arg', 1)
  broadcast_result_index = ('federated_broadcast_result',)
  work_to_before_aggregate = _as_function_of_some_federated_subparameters(
      before_aggregate, [client_data_index, broadcast_result_index]
  )

  # Indices of `work` results in `before_aggregate` result
  aggregate_input_index = ('federated_aggregate_param', 0)
  secure_sum_bitwidth_input_index = ('federated_secure_sum_bitwidth_param', 0)
  secure_sum_input_index = ('federated_secure_sum_param', 0)
  secure_modular_sum_input_index = ('federated_secure_modular_sum_param', 0)
  work_unzipped = federated_language.framework.select_output_from_lambda(
      work_to_before_aggregate,
      [
          aggregate_input_index,
          secure_sum_bitwidth_input_index,
          secure_sum_input_index,
          secure_modular_sum_input_index,
      ],
  )
  work = federated_language.framework.Lambda(
      work_unzipped.parameter_name,
      work_unzipped.parameter_type,
      federated_language.framework.create_federated_zip(work_unzipped.result),
  )
  return compiler.consolidate_and_extract_local_processing(
      work, grappler_config
  )


def _compile_selected_output_to_no_argument_tensorflow(
    comp: federated_language.framework.Lambda,
    path: federated_language.framework.Path,
    grappler_config,
) -> federated_language.framework.CompiledComputation:
  """Compiles the independent value result of `comp` at `path` to TensorFlow."""
  extracted = federated_language.framework.select_output_from_lambda(
      comp, path
  ).result
  return compiler.consolidate_and_extract_local_processing(
      federated_language.framework.Lambda(None, None, extracted),
      grappler_config,
  )


def _compile_selected_output_as_tensorflow_function(
    comp: federated_language.framework.Lambda,
    path: federated_language.framework.Path,
    grappler_config,
) -> federated_language.framework.CompiledComputation:
  """Compiles the functional result of `comp` at `path` to TensorFlow."""
  extracted = federated_language.framework.select_output_from_lambda(
      comp, path
  ).result
  return compiler.consolidate_and_extract_local_processing(
      extracted, grappler_config
  )


def _extract_federated_aggregate_functions(before_aggregate, grappler_config):
  """Extracts federated aggregate functions from `before_aggregate`.

  This function is intended to be used by
  `get_map_reduce_form_for_computation` only. As a result, this function
  does not assert that `before_aggregate` has the expected structure, the
  caller is expected to perform these checks before calling this function.

  Args:
    before_aggregate: The first result of splitting `after_broadcast` on
      aggregate intrinsics.
    grappler_config: An instance of `tf.compat.v1.ConfigProto` to configure
      Grappler graph optimization.

  Returns:
    `zero`, `accumulate`, `merge` and `report` as specified by
    `forms.MapReduceForm`. All are instances of
    `federated_language.framework.CompiledComputation`.

  Raises:
    compiler.MapReduceFormCompilationError: If we extract an ASTs of the wrong
      type.
  """
  federated_aggregate = federated_language.framework.select_output_from_lambda(
      before_aggregate, 'federated_aggregate_param'
  )
  # Index `0` is the value being aggregated.
  zero = _compile_selected_output_to_no_argument_tensorflow(
      federated_aggregate, 1, grappler_config
  )
  accumulate = _compile_selected_output_as_tensorflow_function(
      federated_aggregate, 2, grappler_config
  )
  merge = _compile_selected_output_as_tensorflow_function(
      federated_aggregate, 3, grappler_config
  )
  report = _compile_selected_output_as_tensorflow_function(
      federated_aggregate, 4, grappler_config
  )
  return zero, accumulate, merge, report


def _extract_update(after_aggregate, grappler_config):
  """Extracts `update` from `after_aggregate`.

  This function is intended to be used by
  `get_map_reduce_form_for_computation` only. As a result, this function
  does not assert that `after_aggregate` has the expected structure, the
  caller is expected to perform these checks before calling this function.

  Args:
    after_aggregate: The second result of splitting `after_broadcast` on
      aggregate intrinsics.
    grappler_config: An instance of `tf.compat.v1.ConfigProto` to configure
      Grappler graph optimization.

  Returns:
    `update` as specified by `forms.MapReduceForm`, an instance of
    `federated_language.framework.CompiledComputation`.

  Raises:
    compiler.MapReduceFormCompilationError: If we extract an AST of the wrong
      type.
  """
  after_aggregate_zipped = federated_language.framework.Lambda(
      after_aggregate.parameter_name,
      after_aggregate.parameter_type,
      federated_language.framework.create_federated_zip(after_aggregate.result),
  )
  # `create_federated_zip` doesn't have unique reference names, but we need
  # them for `as_function_of_some_federated_subparameters`.
  after_aggregate_zipped, _ = tree_transformations.uniquify_reference_names(
      after_aggregate_zipped
  )
  server_state_index = ('original_arg', 'original_arg', 0)
  aggregate_result_index = ('intrinsic_results', 'federated_aggregate_result')
  secure_sum_bitwidth_result_index = (
      'intrinsic_results',
      'federated_secure_sum_bitwidth_result',
  )
  secure_sum_result_index = ('intrinsic_results', 'federated_secure_sum_result')
  secure_modular_sum_result_index = (
      'intrinsic_results',
      'federated_secure_modular_sum_result',
  )
  update_with_flat_inputs = _as_function_of_some_federated_subparameters(
      after_aggregate_zipped,
      (
          server_state_index,
          aggregate_result_index,
          secure_sum_bitwidth_result_index,
          secure_sum_result_index,
          secure_modular_sum_result_index,
      ),
  )

  # TODO: b/148942011 - The transformation
  # `zip_selection_as_argument_to_lower_level_lambda` does not support selecting
  # from nested structures, therefore we need to transform the input from
  # <server_state, <aggregation_results...>> into
  # <server_state, aggregation_results...>
  # unpack = <v, <...>> -> <v, ...>
  name_generator = federated_language.framework.unique_name_generator(
      update_with_flat_inputs
  )
  unpack_param_name = next(name_generator)
  original_param_type = update_with_flat_inputs.parameter_type.member  # pytype: disable=attribute-error
  unpack_param_type = federated_language.StructType([
      original_param_type[0],
      federated_language.StructType(original_param_type[1:]),
  ])
  unpack_param_ref = federated_language.framework.Reference(
      unpack_param_name, unpack_param_type
  )
  select = lambda bb, i: federated_language.framework.Selection(bb, index=i)
  unpack = federated_language.framework.Lambda(
      unpack_param_name,
      unpack_param_type,
      federated_language.framework.Struct(
          [select(unpack_param_ref, 0)]
          + [
              select(select(unpack_param_ref, 1), i)
              for i in range(len(original_param_type) - 1)
          ]
      ),
  )

  # update = v -> update_with_flat_inputs(federated_map(unpack, v))
  param_name = next(name_generator)
  param_type = federated_language.FederatedType(
      unpack_param_type, federated_language.SERVER
  )
  param_ref = federated_language.framework.Reference(param_name, param_type)
  update = federated_language.framework.Lambda(
      param_name,
      param_type,
      federated_language.framework.Call(
          update_with_flat_inputs,
          federated_language.framework.create_federated_map_or_apply(
              unpack, param_ref
          ),
      ),
  )
  return compiler.consolidate_and_extract_local_processing(
      update, grappler_config
  )


def _replace_lambda_body_with_call_dominant_form(
    comp: federated_language.framework.Lambda,
) -> federated_language.framework.Lambda:
  """Transforms the body of `comp` to call-dominant form.

  Call-dominant form ensures that all higher-order functions are fully
  resolved, as well that called intrinsics are pulled out into a top-level
  let-binding. This combination of condition ensures first that pattern-matching
  on calls to intrinsics is sufficient to identify communication operators in
  `force_align_and_split_by_intrinsics`, and second that there are no nested
  intrinsics which will cause that function to fail.

  Args:
    comp: `federated_language.framework.Lambda` the body of which to convert to
      call-dominant form.

  Returns:
    A transformed version of `comp`, whose body is call-dominant.
  """
  transformed = transformations.to_call_dominant(comp)
  if not isinstance(transformed, federated_language.framework.Lambda):
    raise federated_language.framework.UnexpectedBlockError(
        federated_language.framework.Lambda, transformed
    )
  return transformed


def _merge_grappler_config_with_default(
    grappler_config: tf.compat.v1.ConfigProto,
) -> tf.compat.v1.ConfigProto:
  py_typecheck.check_type(grappler_config, tf.compat.v1.ConfigProto)
  overridden_grappler_config = tf.compat.v1.ConfigProto()
  overridden_grappler_config.CopyFrom(_GRAPPLER_DEFAULT_CONFIG)
  overridden_grappler_config.MergeFrom(grappler_config)
  return overridden_grappler_config


def get_broadcast_form_for_computation(
    comp: federated_language.framework.ConcreteComputation,
    grappler_config: tf.compat.v1.ConfigProto = _GRAPPLER_DEFAULT_CONFIG,
    *,
    tff_internal_preprocessing: Optional[BuildingBlockFn] = None,
) -> forms.BroadcastForm:
  """Constructs `tff.backends.mapreduce.BroadcastForm` given a computation.

  Args:
    comp: An instance of `federated_language.framework.ConcreteComputation` that
      is compatible with broadcast form. Computations are only compatible if
      they take in a single value placed at server, return a single value placed
      at clients, and do not contain any aggregations.
    grappler_config: An instance of `tf.compat.v1.ConfigProto` to configure
      Grappler graph optimization of the Tensorflow graphs backing the resulting
      `tff.backends.mapreduce.BroadcastForm`. These options are combined with a
      set of defaults that aggressively configure Grappler. If
      `grappler_config_proto` has
      `graph_options.rewrite_options.disable_meta_optimizer=True`, Grappler is
      bypassed.
    tff_internal_preprocessing: An optional function to transform the AST of the
      computation.

  Returns:
    An instance of `tff.backends.mapreduce.BroadcastForm` equivalent to the
    provided `federated_language.Computation`.
  """
  py_typecheck.check_type(
      comp, federated_language.framework.ConcreteComputation
  )
  _check_function_signature_compatible_with_broadcast_form(comp.type_signature)
  py_typecheck.check_type(grappler_config, tf.compat.v1.ConfigProto)
  grappler_config = _merge_grappler_config_with_default(grappler_config)

  bb = comp.to_building_block()
  if tff_internal_preprocessing is not None:
    bb = tff_internal_preprocessing(bb)
  bb, _ = tensorflow_tree_transformations.replace_intrinsics_with_bodies(bb)
  bb = _replace_lambda_body_with_call_dominant_form(bb)

  federated_language.framework.check_contains_only_reducible_intrinsics(bb)
  aggregations = federated_language.framework.find_aggregations_in_tree(bb)
  if aggregations:
    raise ValueError(
        '`get_broadcast_form_for_computation` called with computation'
        f' containing {len(aggregations)} aggregations, but broadcast form does'
        ' not allow aggregation. Full list of aggregations:\n{aggregations}'
    )

  before_broadcast, after_broadcast = _split_ast_on_broadcast(bb)
  compute_server_context = _extract_compute_server_context(
      before_broadcast, grappler_config
  )
  client_processing = _extract_client_processing(
      after_broadcast, grappler_config
  )

  def _create_comp(proto):
    return federated_language.framework.ConcreteComputation(
        computation_proto=proto,
        context_stack=federated_language.framework.get_context_stack(),
    )

  compute_server_context, client_processing = (
      _create_comp(bb.to_proto())
      for bb in (compute_server_context, client_processing)
  )

  comp_param_names = [n for n, _ in comp.type_signature.parameter.items()]  # pytype: disable=attribute-error
  server_data_label, client_data_label = comp_param_names
  return forms.BroadcastForm(
      compute_server_context,
      client_processing,
      server_data_label=server_data_label,
      client_data_label=client_data_label,
  )


def get_map_reduce_form_for_computation(
    comp: federated_language.framework.ConcreteComputation,
    grappler_config: tf.compat.v1.ConfigProto = _GRAPPLER_DEFAULT_CONFIG,
    *,
    tff_internal_preprocessing: Optional[BuildingBlockFn] = None,
) -> forms.MapReduceForm:
  """Constructs `tff.backends.mapreduce.MapReduceForm` for a computation.

  Args:
    comp: An instance of `federated_language.framework.ConcreteComputation` that
      is compatible with MapReduce form. The computation must take exactly two
      arguments, and the first must be a state value placed at `SERVER`. The
      computation must return exactly two values. The type of the first element
      in the result must also be assignable to the first element of the
      parameter.
    grappler_config: An optional instance of `tf.compat.v1.ConfigProto` to
      configure Grappler graph optimization of the TensorFlow graphs backing the
      resulting `tff.backends.mapreduce.MapReduceForm`. These options are
      combined with a set of defaults that aggressively configure Grappler. If
      the input `grappler_config` has
      `graph_options.rewrite_options.disable_meta_optimizer=True`, Grappler is
      bypassed.
    tff_internal_preprocessing: An optional function to transform the AST of the
      iterative process.

  Returns:
    An instance of `tff.backends.mapreduce.MapReduceForm` equivalent to the
    provided `federated_language.framework.ConcreteComputation`.

  Raises:
    TypeError: If the arguments are of the wrong types.
    compiler.MapReduceFormCompilationError: If the compilation process fails.
  """
  py_typecheck.check_type(
      comp, federated_language.framework.ConcreteComputation
  )
  comp_bb = check_computation_compatible_with_map_reduce_form(
      comp, tff_internal_preprocessing=tff_internal_preprocessing
  )
  py_typecheck.check_type(grappler_config, tf.compat.v1.ConfigProto)
  grappler_config = _merge_grappler_config_with_default(grappler_config)

  comp_bb, _ = tree_transformations.uniquify_reference_names(comp_bb)
  before_broadcast, after_broadcast = _split_ast_on_broadcast(comp_bb)
  before_aggregate, after_aggregate = _split_ast_on_aggregate(after_broadcast)

  prepare = _extract_prepare(before_broadcast, grappler_config)
  work = _extract_work(before_aggregate, grappler_config)
  zero, accumulate, merge, report = _extract_federated_aggregate_functions(
      before_aggregate, grappler_config
  )
  secure_sum_bitwidth = _compile_selected_output_to_no_argument_tensorflow(
      before_aggregate,
      ('federated_secure_sum_bitwidth_param', 1),
      grappler_config,
  )
  secure_sum_max_input = _compile_selected_output_to_no_argument_tensorflow(
      before_aggregate, ('federated_secure_sum_param', 1), grappler_config
  )
  secure_sum_modulus = _compile_selected_output_to_no_argument_tensorflow(
      before_aggregate,
      ('federated_secure_modular_sum_param', 1),
      grappler_config,
  )
  update = _extract_update(after_aggregate, grappler_config)

  def _create_comp(proto):
    return federated_language.framework.ConcreteComputation(
        computation_proto=proto,
        context_stack=federated_language.framework.get_context_stack(),
    )

  blocks = (
      prepare,
      work,
      zero,
      accumulate,
      merge,
      report,
      secure_sum_bitwidth,
      secure_sum_max_input,
      secure_sum_modulus,
      update,
  )
  comps = [_create_comp(bb.to_proto()) for bb in blocks]
  return forms.MapReduceForm(comp.type_signature, *comps)


def get_distribute_aggregate_form_for_computation(
    comp: federated_language.framework.ConcreteComputation,
    *,
    tff_internal_preprocessing: Optional[BuildingBlockFn] = None,
) -> forms.DistributeAggregateForm:
  """Constructs `DistributeAggregateForm` for a computation.

  Args:
    comp: An instance of `federated_language.framework.ConcreteComputation` that
      is compatible with `DistributeAggregateForm`. The computation must take
      exactly two arguments, and the first must be a state value placed at
      `SERVER`. The computation must return exactly two values. The type of the
      first element in the result must also be assignable to the first element
      of the parameter.
    tff_internal_preprocessing: An optional function to transform the AST of the
      iterative process.

  Returns:
    An instance of `tff.backends.mapreduce.DistributeAggregateForm` equivalent
    to the provided `federated_language.framework.Computation`.

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  py_typecheck.check_type(comp, federated_language.framework.Computation)

  # Apply any requested preprocessing to the computation.
  comp_tree = comp.to_building_block()
  if tff_internal_preprocessing is not None:
    comp_tree = tff_internal_preprocessing(comp_tree)

  # Check that the computation has the expected structure.
  comp_type = comp_tree.type_signature
  _check_type_is_fn(comp_type, '`comp`', TypeError)
  if (
      not isinstance(comp_type.parameter, federated_language.StructType)
      or len(comp_type.parameter) != 2
  ):  # pytype: disable=attribute-error
    raise TypeError(
        'Expected `comp` to take two arguments, found parameter '
        f' type:\n{comp_type.parameter}'  # pytype: disable=attribute-error
    )
  if (
      not isinstance(comp_type.result, federated_language.StructType)
      or len(comp_type.result) != 2
  ):  # pytype: disable=attribute-error
    raise TypeError(
        'Expected `comp` to return two values, found result '
        f'type:\n{comp_type.result}'  # pytype: disable=attribute-error
    )
  if not isinstance(comp_tree, federated_language.framework.Lambda):
    raise federated_language.framework.UnexpectedBlockError(
        federated_language.framework.Lambda, comp_tree
    )
  comp_tree = _replace_lambda_body_with_call_dominant_form(comp_tree)
  comp_tree, _ = tree_transformations.uniquify_reference_names(comp_tree)
  federated_language.framework.check_broadcast_not_dependent_on_aggregate(
      comp_tree
  )

  # To generate the DistributeAggregateForm for the computation, we will split
  # the computation twice, first on broadcast intrinsics and then on aggregation
  # intrinsics. To ensure that any non-client-placed unbound refs used by the
  # aggregation intrinsics args are fed via the temporary state (as opposed to
  # via the clients, which shouldn't be returning non-client-placed data), add a
  # special broadcast call to the computation that depends on these args (if any
  # exist). Examples of these args are the `modulus` for a
  # federated_secure_modular_sum call or the `zero` for a federated_aggregate
  # call. Note that this injected broadcast call will not actually broadcast
  # these args to the clients (it broadcasts an empty struct). The sole purpose
  # of the injected broadcast is to establish a dependency that forces the calls
  # associated with these non-client-placed refs to appear in the *first* part
  # of the split on broadcast intrinsics rather than potentially appearing in
  # the *last* part of the split on broadcast intrinsics.
  args_needing_broadcast_dependency = []
  unbound_refs = federated_language.framework.get_map_of_unbound_references(
      comp_tree
  )

  def _find_non_client_placed_args(inner_comp):
    # Examine the args of the aggregation intrinsic calls.
    if (
        isinstance(inner_comp, federated_language.framework.Call)
        and isinstance(
            inner_comp.function, federated_language.framework.Intrinsic
        )
        and inner_comp.function.intrinsic_def().aggregation_kind
    ):
      if isinstance(inner_comp.argument, federated_language.framework.Struct):
        aggregation_args = inner_comp.argument
      else:
        aggregation_args = [inner_comp.argument]
      unbound_ref_names_for_intrinsic = unbound_refs[inner_comp.argument]

      for aggregation_arg in aggregation_args:
        if unbound_refs[aggregation_arg].issubset(
            unbound_ref_names_for_intrinsic
        ):
          # If the arg is non-placed or server-placed, prepare to create a
          # federated broadcast that depends on it by normalizing it to a
          # server-placed value.
          if not isinstance(
              aggregation_arg.type_signature, federated_language.FederatedType
          ):

            def _has_placement(type_spec):
              return isinstance(
                  type_spec.type_signature, federated_language.FederatedType
              )

            if (
                federated_language.framework.computation_count(
                    aggregation_arg, _has_placement
                )
                > 0
            ):
              raise TypeError(
                  'DistributeAggregateForm cannot handle an aggregation '
                  f'intrinsic arg with type {aggregation_arg.type_signature}'
              )
            args_needing_broadcast_dependency.append(
                federated_language.framework.create_federated_value(
                    aggregation_arg, federated_language.SERVER
                )
            )
          elif (
              aggregation_arg.type_signature.placement
              == federated_language.SERVER
          ):
            args_needing_broadcast_dependency.append(aggregation_arg)

      return inner_comp, True
    return inner_comp, False

  federated_language.framework.visit_preorder(
      comp_tree, _find_non_client_placed_args
  )

  # Add an injected broadcast call to the computation that depends on the
  # identified non-client-placed args, if any exist. To avoid broadcasting the
  # actual non-client-placed args (undesirable from both a privacy and
  # efficiency standpoint), instead broadcast the empty struct result generated
  # by an intermediate map call that takes the non-client-placed args as input.
  # This approach should work as long as the intermediate map call does not get
  # pruned by various tree transformations. Currently, tree transformations
  # such as to_call_dominant do not recognize that our intermediate map call
  # here can be drastically simplified.
  if args_needing_broadcast_dependency:
    zipped_args_needing_broadcast_dependency = (
        federated_language.framework.create_federated_zip(
            federated_language.framework.Struct(
                args_needing_broadcast_dependency
            )
        )
    )
    injected_broadcast = federated_language.framework.create_federated_broadcast(
        federated_language.framework.create_federated_apply(
            federated_language.framework.Lambda(
                'ignored_param',
                zipped_args_needing_broadcast_dependency.type_signature.member,
                federated_language.framework.Struct([]),
            ),
            zipped_args_needing_broadcast_dependency,
        )
    )
    # Add the injected broadcast call to the block locals.
    revised_block_locals = comp_tree.result.locals + [(
        'injected_broadcast_ref',
        injected_broadcast,
    )]
    # Add a reference to the injected broadcast call in the result so that it
    # does not get pruned by various tree transformations. We will remove this
    # additional element in the result after the first split operation.
    revised_block_result = list(comp_tree.result.result.items()) + [
        federated_language.framework.Reference(
            'injected_broadcast_ref',
            injected_broadcast.type_signature,
        )
    ]
    comp_tree = federated_language.framework.Lambda(
        comp_tree.parameter_name,
        comp_tree.parameter_type,
        federated_language.framework.Block(
            revised_block_locals,
            federated_language.framework.Struct(revised_block_result),
        ),
    )

  # Split first on the broadcast intrinsics.
  # - The "before" comp in this split (which will eventually become the
  # server_prepare portion of the DAF) should only depend on the server portion
  # of the original comp input.
  # - The "intrinsic" comp in this split (which will eventually become the
  # server_to_client_broadcast portion of the DAF) should not depend on any
  # portion of the original comp.
  # - The "after" comp in this split (which will eventually become the
  # client_work, client_to_server_aggregation, and server_result portions of
  # the DAF) should be allowed to depend only the client portion of the original
  # comp. Any server-related or non-placed dependencies will be passed via the
  # intermediate state.
  server_state_index = 0
  client_data_index = 1
  server_prepare, server_to_client_broadcast, after_broadcast = (
      transformations.divisive_force_align_and_split_by_intrinsics(
          comp_tree,
          federated_language.framework.get_broadcast_intrinsics(),
          before_comp_allowed_original_arg_subparameters=[(
              server_state_index,
          )],
          intrinsic_comp_allowed_original_arg_subparameters=[],
          after_comp_allowed_original_arg_subparameters=[(client_data_index,)],
      )
  )

  # Helper method to replace a lambda with parameter that is a single-element
  # struct with a lambda that uses the element directly.
  def _unnest_lambda_parameter(comp):
    assert isinstance(comp, federated_language.framework.Lambda)
    assert isinstance(comp.parameter_type, federated_language.StructType)

    name_generator = federated_language.framework.unique_name_generator(comp)
    new_param_name = next(name_generator)
    replacement_ref = federated_language.framework.Reference(
        new_param_name, comp.parameter_type[0]
    )
    modified_comp_body = tree_transformations.replace_selections(
        comp.result, comp.parameter_name, {(0,): replacement_ref}
    )
    modified_comp = federated_language.framework.Lambda(
        replacement_ref.name, replacement_ref.type_signature, modified_comp_body
    )
    federated_language.framework.check_contains_no_unbound_references(
        modified_comp
    )

    return modified_comp

  # Finalize the server_prepare and server_to_client_broadcast comps by
  # removing a layer of nesting from the input parameter.
  server_prepare = _unnest_lambda_parameter(server_prepare)
  server_to_client_broadcast = _unnest_lambda_parameter(
      server_to_client_broadcast
  )

  # Next split on the aggregation intrinsics. If an injected broadcast was used
  # in the previous step, drop the last element in the output (which held the
  # output of the injected broadcast) before performing the second split.
  # - The "before" comp in this split (which will eventually become the
  # client_work portion of the DAF) should only depend on the client portion of
  # the original comp input (i.e. the client portion of the "original_arg"
  # portion of the after_broadcast input) and the portion of the after_broadcast
  # input that represents the intrinsic output that was produced in the first
  # split (i.e. the broadcast output).
  # - The "intrinsic" comp in this split (which will eventually become the
  # client_to_server_aggregation portion of the DAF) should only depend on the
  # portion of the after_broadcast input that represents the intermediate state
  # that was produced in the first split.
  # - The "after" comp in this split (which will eventually become the
  # server_result portion of the DAF) should only depend on the server portion
  # of the original comp input (i.e. the server portion of the "original_arg"
  # portion of the after_broadcast input) and the portion of the after_broadcast
  # input that represents the intermediate state that was produced in the first
  # split.
  if args_needing_broadcast_dependency:
    assert isinstance(
        after_broadcast.result.result, federated_language.framework.Struct
    )
    # Check that the last element of the result is the expected empty struct
    # associated with the injected broadcast call.
    result_len = len(after_broadcast.result.result)
    injected_broadcast_result = after_broadcast.result.result[result_len - 1]
    assert isinstance(
        injected_broadcast_result.type_signature.member,
        federated_language.StructType,
    )
    assert not injected_broadcast_result.type_signature.member
    after_broadcast = federated_language.framework.Lambda(
        after_broadcast.parameter_name,
        after_broadcast.parameter_type,
        federated_language.framework.Block(
            after_broadcast.result.locals,
            federated_language.framework.Struct(
                list(after_broadcast.result.result.items())[:-1]
            ),
        ),
    )
  client_data_index_in_after_broadcast_param = 0
  intrinsic_results_index_in_after_broadcast_param = 1
  intermediate_state_index_in_after_broadcast_param = 2
  client_work, client_to_server_aggregation, server_result = (
      transformations.divisive_force_align_and_split_by_intrinsics(
          after_broadcast,
          federated_language.framework.get_aggregation_intrinsics(),
          before_comp_allowed_original_arg_subparameters=[
              (client_data_index_in_after_broadcast_param,),
              (intrinsic_results_index_in_after_broadcast_param,),
          ],
          intrinsic_comp_allowed_original_arg_subparameters=[(
              intermediate_state_index_in_after_broadcast_param,
          )],
          after_comp_allowed_original_arg_subparameters=[
              (intermediate_state_index_in_after_broadcast_param,),
          ],
      )
  )

  # Drop the intermediate_state produced by the second split that is part of
  # the client_work output.
  index_of_intrinsic_args_in_client_work_result = 0
  client_work = federated_language.framework.select_output_from_lambda(
      client_work, index_of_intrinsic_args_in_client_work_result
  )

  # Drop the intermediate_state param produced by the second split that is part
  # of the server_result parameter (but keep the part of the param that
  # corresponds to the intermediate_state produced by the first split).
  intermediate_state_index_in_server_result_param = 0
  aggregation_result_index_in_server_result_param = 1
  server_result = tree_transformations.as_function_of_some_subparameters(
      server_result,
      [
          (intermediate_state_index_in_server_result_param,),
          (aggregation_result_index_in_server_result_param,),
      ],
  )

  blocks = (
      server_prepare,
      server_to_client_broadcast,
      client_work,
      client_to_server_aggregation,
      server_result,
  )

  def _create_comp(proto):
    return federated_language.framework.ConcreteComputation(
        computation_proto=proto,
        context_stack=federated_language.framework.get_context_stack(),
    )

  comps = [_create_comp(bb.to_proto()) for bb in blocks]

  return forms.DistributeAggregateForm(comp.type_signature, *comps)

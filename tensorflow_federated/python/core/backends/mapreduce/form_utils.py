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

Note: Refer to `get_iterative_process_for_map_reduce_form()` for the meaning of
variable names used in this module.
"""

from typing import Callable, Dict, List, Tuple

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.backends.mapreduce import forms
from tensorflow_federated.python.core.backends.mapreduce import transformations
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_reductions
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import transformations as compiler_transformations
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper_instances
from tensorflow_federated.python.core.templates import iterative_process

_GRAPPLER_DEFAULT_CONFIG = tf.compat.v1.ConfigProto()
_AGGRESSIVE = _GRAPPLER_DEFAULT_CONFIG.graph_options.rewrite_options.AGGRESSIVE
_GRAPPLER_DEFAULT_CONFIG.graph_options.rewrite_options.memory_optimization = _AGGRESSIVE
_GRAPPLER_DEFAULT_CONFIG.graph_options.rewrite_options.constant_folding = _AGGRESSIVE
_GRAPPLER_DEFAULT_CONFIG.graph_options.rewrite_options.arithmetic_optimization = _AGGRESSIVE
_GRAPPLER_DEFAULT_CONFIG.graph_options.rewrite_options.loop_optimization = _AGGRESSIVE
_GRAPPLER_DEFAULT_CONFIG.graph_options.rewrite_options.function_optimization = _AGGRESSIVE


def get_computation_for_broadcast_form(
    bf: forms.BroadcastForm) -> computation_base.Computation:
  """Creates `tff.Computation` from a broadcast form."""
  py_typecheck.check_type(bf, forms.BroadcastForm)
  server_data_type = bf.compute_server_context.type_signature.parameter
  client_data_type = bf.client_processing.type_signature.parameter[1]
  comp_parameter_type = computation_types.StructType([
      (bf.server_data_label, computation_types.at_server(server_data_type)),
      (bf.client_data_label, computation_types.at_clients(client_data_type)),
  ])

  @computations.federated_computation(comp_parameter_type)
  def computation(arg):
    server_data, client_data = arg
    context_at_server = intrinsics.federated_map(bf.compute_server_context,
                                                 server_data)
    context_at_clients = intrinsics.federated_broadcast(context_at_server)
    client_processing_arg = intrinsics.federated_zip(
        (context_at_clients, client_data))
    return intrinsics.federated_map(bf.client_processing, client_processing_arg)

  return computation


def get_iterative_process_for_map_reduce_form(
    mrf: forms.MapReduceForm) -> iterative_process.IterativeProcess:
  """Creates `tff.templates.IterativeProcess` from a MapReduce form.

  Args:
    mrf: An instance of `tff.backends.mapreduce.MapReduceForm`.

  Returns:
    An instance of `tff.templates.IterativeProcess` that corresponds to `mrf`.

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  py_typecheck.check_type(mrf, forms.MapReduceForm)

  @computations.federated_computation
  def init_computation():
    return intrinsics.federated_value(mrf.initialize(), placements.SERVER)

  next_parameter_type = computation_types.StructType([
      (mrf.server_state_label, init_computation.type_signature.result),
      (mrf.client_data_label,
       computation_types.FederatedType(mrf.work.type_signature.parameter[0],
                                       placements.CLIENTS)),
  ])

  @computations.federated_computation(next_parameter_type)
  def next_computation(arg):
    """The logic of a single MapReduce processing round."""
    s1 = arg[0]
    c1 = arg[1]
    s2 = intrinsics.federated_map(mrf.prepare, s1)
    c2 = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([c1, c2])
    c4 = intrinsics.federated_map(mrf.work, c3)
    c5 = c4[0]
    c6 = c4[1]
    s3 = intrinsics.federated_aggregate(c5, mrf.zero(), mrf.accumulate,
                                        mrf.merge, mrf.report)
    s4 = intrinsics.federated_secure_sum_bitwidth(c6, mrf.bitwidth())
    s5 = intrinsics.federated_zip([s3, s4])
    s6 = intrinsics.federated_zip([s1, s5])
    s7 = intrinsics.federated_map(mrf.update, s6)
    s8 = s7[0]
    s9 = s7[1]
    return s8, s9

  return iterative_process.IterativeProcess(init_computation, next_computation)


def _check_len(
    target,
    length,
    err_fn: Callable[[str],
                     Exception] = transformations.MapReduceFormCompilationError,
):
  py_typecheck.check_type(length, int)
  if len(target) != length:
    raise err_fn('Expected length of {}, found {}.'.format(length, len(target)))


def _check_placement(
    target,
    placement: placements.PlacementLiteral,
    err_fn: Callable[[str],
                     Exception] = transformations.MapReduceFormCompilationError,
):
  py_typecheck.check_type(target, computation_types.FederatedType)
  py_typecheck.check_type(placement, placements.PlacementLiteral)
  if target.placement != placement:
    raise err_fn(
        'Expected value with placement {}, found value of type {}.'.format(
            placement, target))


def _check_type_equal(
    actual,
    expected,
    err_fn: Callable[[str],
                     Exception] = transformations.MapReduceFormCompilationError,
):
  py_typecheck.check_type(actual, computation_types.Type)
  py_typecheck.check_type(expected, computation_types.Type)
  if not actual.is_equivalent_to(expected):
    raise err_fn('Expected type of {}, found {}.'.format(expected, actual))


def _check_type(
    target,
    type_spec,
    err_fn: Callable[[str],
                     Exception] = transformations.MapReduceFormCompilationError,
):
  py_typecheck.check_type(type_spec, type)
  if not isinstance(target, type_spec):
    raise err_fn('Expected type of {}, found {}.'.format(
        type_spec, type(target)))


def _check_type_is_fn(
    target: computation_types.Type,
    name: str,
    err_fn: Callable[[str],
                     Exception] = transformations.MapReduceFormCompilationError,
):
  if not target.is_function():
    raise err_fn(f'Expected {name} to be a function, but {name} had type '
                 f'{target}.')


def _check_type_is_no_arg_fn(
    target: computation_types.Type,
    name: str,
    err_fn: Callable[[str],
                     Exception] = transformations.MapReduceFormCompilationError,
):
  _check_type_is_fn(target, name, err_fn)
  if target.parameter is not None:
    raise err_fn(f'Expected {name} to take no argument, but found '
                 f'parameter of type {target.parameter}.')


def _check_function_signature_compatible_with_broadcast_form(
    function_type: computation_types.FunctionType):
  """Tests compatibility with `tff.backends.mapreduce.BroadcastForm`."""
  py_typecheck.check_type(function_type, computation_types.FunctionType)
  if not (function_type.parameter.is_struct() and
          len(function_type.parameter) == 2):
    raise TypeError(
        '`BroadcastForm` requires a computation which accepts two arguments '
        '(server data and client data) but found parameter type:\n'
        f'{function_type.parameter}')
  server_data_type, client_data_type = function_type.parameter
  if not (server_data_type.is_federated() and
          server_data_type.placement.is_server()):
    raise TypeError(
        '`BroadcastForm` expects a computation whose first parameter is server '
        'data (a federated type placed at server) but found first parameter of '
        f'type:\n{server_data_type}')
  if not (client_data_type.is_federated() and
          client_data_type.placement.is_clients()):
    raise TypeError(
        '`BroadcastForm` expects a computation whose first parameter is client '
        'data (a federated type placed at clients) but found first parameter '
        f'of type:\n{client_data_type}')
  result_type = function_type.result
  if not (result_type.is_federated() and result_type.placement.is_clients()):
    raise TypeError(
        '`BroadcastForm` expects a computation whose result is client data '
        '(a federated type placed at clients) but found result type:\n'
        f'{result_type}')


def check_iterative_process_compatible_with_map_reduce_form(
    ip: iterative_process.IterativeProcess):
  """Tests compatibility with `tff.backends.mapreduce.MapReduceForm`.

  Note: the conditions here are specified in the documentation for
    `get_map_reduce_form_for_iterative_process`. Changes to this function should
    be propagated to that documentation.

  Args:
    ip: An instance of `tff.templates.IterativeProcess` to check for
      compatibility with `tff.backends.mapreduce.MapReduceForm`.

  Returns:
    TFF-internal building-blocks representing the validated and simplified
    `initialize` and `next` computations.

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  py_typecheck.check_type(ip, iterative_process.IterativeProcess)
  initialize_tree = ip.initialize.to_building_block()
  next_tree = ip.next.to_building_block()

  init_type = initialize_tree.type_signature
  _check_type_is_no_arg_fn(init_type, '`initialize`', TypeError)
  if (not init_type.result.is_federated() or
      init_type.result.placement != placements.SERVER):
    raise TypeError('Expected `initialize` to return a single federated value '
                    'placed at server (type `T@SERVER`), found return type:\n'
                    f'{init_type.result}')

  next_type = next_tree.type_signature
  _check_type_is_fn(next_type, '`next`', TypeError)
  if not next_type.parameter.is_struct() or len(next_type.parameter) != 2:
    raise TypeError('Expected `next` to take two arguments, found parameter '
                    f' type:\n{next_type.parameter}')
  if not next_type.result.is_struct() or len(next_type.result) != 2:
    raise TypeError('Expected `next` to return two values, found result '
                    f'type:\n{next_type.result}')

  initialize_tree, _ = intrinsic_reductions.replace_intrinsics_with_bodies(
      initialize_tree)
  next_tree, _ = intrinsic_reductions.replace_intrinsics_with_bodies(next_tree)
  next_tree = _replace_lambda_body_with_call_dominant_form(next_tree)

  tree_analysis.check_contains_only_reducible_intrinsics(initialize_tree)
  tree_analysis.check_contains_only_reducible_intrinsics(next_tree)
  tree_analysis.check_broadcast_not_dependent_on_aggregate(next_tree)

  return initialize_tree, next_tree


def _untuple_broadcast_only_before_after(before, after):
  """Removes the tuple-ing of the `broadcast` params and results."""
  # Since there is only a single intrinsic here, there's no need for the outer
  # `{intrinsic_name}_param`/`{intrinsic_name}_result` tuples.
  untupled_before = transformations.select_output_from_lambda(
      before, 'federated_broadcast_param')
  after_param_name = next(building_block_factory.unique_name_generator(after))
  after_param_type = computation_types.StructType([
      ('original_arg', after.parameter_type.original_arg),
      ('federated_broadcast_result',
       after.parameter_type.intrinsic_results.federated_broadcast_result),
  ])
  after_param_ref = building_blocks.Reference(after_param_name,
                                              after_param_type)
  untupled_after = building_blocks.Lambda(
      after_param_name, after_param_type,
      building_blocks.Call(
          after,
          building_blocks.Struct([
              ('original_arg',
               building_blocks.Selection(after_param_ref, 'original_arg')),
              ('intrinsic_results',
               building_blocks.Struct([
                   ('federated_broadcast_result',
                    building_blocks.Selection(after_param_ref,
                                              'federated_broadcast_result'))
               ]))
          ])))
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
      bb, [building_block_factory.create_null_federated_broadcast()])
  return _untuple_broadcast_only_before_after(before, after)


def _split_ast_on_aggregate(bb):
  """Splits an AST on reduced aggregation intrinsics.

  Args:
    bb: An AST containing `federated_aggregate` or
      `federated_secure_sum_bitwidth` aggregations.

  Returns:
    Two ASTs, the first of which maps comp's input to the arguments
    to `federated_aggregate` and `federated_secure_sum_bitwidth`, and the
    second of which maps comp's input and the output of `federated_aggregate`
    and `federated_secure_sum_bitwidth` to comp's output.
  """
  return transformations.force_align_and_split_by_intrinsics(
      bb, [
          building_block_factory.create_null_federated_aggregate(),
          building_block_factory.create_null_federated_secure_sum_bitwidth()
      ])


def _prepare_for_rebinding(bb):
  """Replaces `bb` with semantically equivalent version for rebinding."""
  all_equal_normalized = transformations.normalize_all_equal_bit(bb)
  identities_removed, _ = tree_transformations.remove_mapped_or_applied_identity(
      all_equal_normalized)
  for_rebind, _ = compiler_transformations.prepare_for_rebinding(
      identities_removed)
  return for_rebind


def _construct_selection_from_federated_tuple(
    federated_tuple: building_blocks.ComputationBuildingBlock, index: int,
    name_generator) -> building_blocks.ComputationBuildingBlock:
  """Selects the index `selected_index` from `federated_tuple`."""
  federated_tuple.type_signature.check_federated()
  member_type = federated_tuple.type_signature.member
  member_type.check_struct()
  param_name = next(name_generator)
  selecting_function = building_blocks.Lambda(
      param_name, member_type,
      building_blocks.Selection(
          building_blocks.Reference(param_name, member_type),
          index=index,
      ))
  return building_block_factory.create_federated_map_or_apply(
      selecting_function, federated_tuple)


def _replace_selections(
    bb: building_blocks.ComputationBuildingBlock,
    ref_name: str,
    path_to_replacement: Dict[Tuple[int, ...],
                              building_blocks.ComputationBuildingBlock],
) -> building_blocks.ComputationBuildingBlock:
  """Identifies selection pattern and replaces with new binding.

  Note that this function is somewhat brittle in that it only replaces AST
  fragments of exactly the form `ref_name[i][j][k]` (for path `(i, j, k)`).
  That is, it will not detect `let x = ref_name[i][j] in x[k]` or similar.

  This is only sufficient because, at the point this function has been called,
  called lambdas have been replaced with blocks and blocks have been inlined,
  so there are no reference chains that must be traced back. Any reference which
  would eventually resolve to a part of a lambda's parameter instead refers to
  the parameter directly. Similarly, selections from tuples have been collapsed.
  The remaining concern would be selections via calls to opaque compiled
  compuations, which we error on.

  Args:
    bb: Instance of `building_blocks.ComputationBuildingBlock` in which we wish
      to replace the selections from reference `ref_name` with any path in
      `paths_to_replacement` with the corresponding building block.
    ref_name: Name of the reference to look for selectiosn from.
    path_to_replacement: A map from selection path to the building block with
      which to replace the selection. Note; it is not valid to specify
      overlapping selection paths (where one path encompasses another).

  Returns:
    A possibly transformed version of `bb` with nodes matching the
    selection patterns replaced.
  """

  def _replace(inner_bb):
    # Start with an empty selection
    path = []
    selection = inner_bb
    while selection.is_selection():
      path.append(selection.as_index())
      selection = selection.source
    # In ASTs like x[0][1], we'll see the last (outermost) selection first.
    path.reverse()
    path = tuple(path)
    if (selection.is_reference() and selection.name == ref_name and
        path in path_to_replacement):
      return path_to_replacement[path], True
    if (inner_bb.is_call() and inner_bb.function.is_compiled_computation() and
        inner_bb.argument is not None and inner_bb.argument.is_reference() and
        inner_bb.argument.name == ref_name):
      raise ValueError('Encountered called graph on reference pattern in TFF '
                       'AST; this means relying on pattern-matching when '
                       'rebinding arguments may be insufficient. Ensure that '
                       'arguments are rebound before decorating references '
                       'with called identity graphs.')
    return inner_bb, False

  result, _ = transformation_utils.transform_postorder(bb, _replace)
  return result


def _as_function_of_single_subparameter(bb: building_blocks.Lambda,
                                        index: int) -> building_blocks.Lambda:
  """Turns `x -> ...only uses x_i...` into `x_i -> ...only uses x_i`."""
  tree_analysis.check_has_unique_names(bb)
  bb = _prepare_for_rebinding(bb)
  new_name = next(building_block_factory.unique_name_generator(bb))
  new_ref = building_blocks.Reference(new_name,
                                      bb.type_signature.parameter[index])
  new_lambda_body = _replace_selections(bb.result, bb.parameter_name,
                                        {(index,): new_ref})
  new_lambda = building_blocks.Lambda(new_ref.name, new_ref.type_signature,
                                      new_lambda_body)
  tree_analysis.check_contains_no_new_unbound_references(bb, new_lambda)
  return new_lambda


class _ParameterSelectionError(TypeError):
  pass


class _NonFederatedSelectionError(TypeError):
  pass


class _MismatchedSelectionPlacementError(TypeError):
  pass


def _as_function_of_some_federated_subparameters(
    bb: building_blocks.Lambda,
    paths: List[Tuple[int, ...]],
) -> building_blocks.Lambda:
  """Turns `x -> ...only uses parts of x...` into `parts_of_x -> ...`."""
  tree_analysis.check_has_unique_names(bb)
  bb = _prepare_for_rebinding(bb)
  name_generator = building_block_factory.unique_name_generator(bb)

  type_list = []
  for path in paths:
    selected_type = bb.parameter_type
    for index in path:
      if not (selected_type.is_struct() and len(selected_type) > index):
        raise _ParameterSelectionError(
            'Attempted to rebind references to parameter selection path '
            f'{path}, which is not a valid selection from type '
            f'{bb.parameter_type}. Original AST:\n{bb}')
      selected_type = selected_type[index]
    if not selected_type.is_federated():
      raise _NonFederatedSelectionError(
          'Attempted to rebind references to parameter selection path '
          f'{path} from type {bb.parameter_type}, but the value at that path '
          f'was of non-federated type {selected_type}. Selections must all '
          f'be of federated type. Original AST:\n{bb}')

    type_list.append(selected_type)

  placement = type_list[0].placement
  if not all(x.placement is placement for x in type_list):
    raise _MismatchedSelectionPlacementError(
        'In order to zip the argument to the lower-level lambda together, all '
        'selected arguments should be at the same placement. Your selections '
        f'have resulted in the list of types:\n{type_list}')

  zip_type = computation_types.FederatedType([x.member for x in type_list],
                                             placement=placement)
  ref_to_zip = building_blocks.Reference(next(name_generator), zip_type)
  path_to_replacement = {}
  for i, path in enumerate(paths):
    path_to_replacement[path] = _construct_selection_from_federated_tuple(
        ref_to_zip, i, name_generator)

  new_lambda_body = _replace_selections(bb.result, bb.parameter_name,
                                        path_to_replacement)
  lambda_with_zipped_param = building_blocks.Lambda(ref_to_zip.name,
                                                    ref_to_zip.type_signature,
                                                    new_lambda_body)
  tree_analysis.check_contains_no_new_unbound_references(
      bb, lambda_with_zipped_param)

  return lambda_with_zipped_param


def _extract_compute_server_context(before_broadcast, grappler_config):
  """Extracts `compute_server_config` from `before_broadcast`."""
  server_data_index_in_before_broadcast = 0
  compute_server_context = _as_function_of_single_subparameter(
      before_broadcast, server_data_index_in_before_broadcast)
  return transformations.consolidate_and_extract_local_processing(
      compute_server_context, grappler_config)


def _extract_client_processing(after_broadcast, grappler_config):
  """Extracts `client_processing` from `after_broadcast`."""
  context_from_server_index_in_after_broadcast = (1,)
  client_data_index_in_after_broadcast = (0, 1)
  # NOTE: the order of parameters here is different from `work`.
  # `work` is odd in that it takes its parameters as `(data, params)` rather
  # than `(params, data)` (the order of the iterative process / computation).
  # Here, we use the same `(params, data)` ordering as in the input computation.
  client_processing = _as_function_of_some_federated_subparameters(
      after_broadcast, [
          context_from_server_index_in_after_broadcast,
          client_data_index_in_after_broadcast
      ])
  return transformations.consolidate_and_extract_local_processing(
      client_processing, grappler_config)


def _extract_prepare(before_broadcast, grappler_config):
  """extracts `prepare` from `before_broadcast`.

  This function is intended to be used by
  `get_map_reduce_form_for_iterative_process` only. As a result, this function
  does not assert that `before_broadcast` has the expected structure, the
  caller is expected to perform these checks before calling this function.

  Args:
    before_broadcast: The first result of splitting `next_bb` on
      `intrinsic_defs.FEDERATED_BROADCAST`.
    grappler_config: An instance of `tf.compat.v1.ConfigProto` to configure
      Grappler graph optimization.

  Returns:
    `prepare` as specified by `forms.MapReduceForm`, an instance of
    `building_blocks.CompiledComputation`.

  Raises:
    transformations.MapReduceFormCompilationError: If we extract an AST of the
      wrong type.
  """
  s1_index_in_before_broadcast = 0
  s1_to_s2_computation = _as_function_of_single_subparameter(
      before_broadcast, s1_index_in_before_broadcast)
  return transformations.consolidate_and_extract_local_processing(
      s1_to_s2_computation, grappler_config)


def _extract_work(before_aggregate, grappler_config):
  """Extracts `work` from `before_aggregate`.

  This function is intended to be used by
  `get_map_reduce_form_for_iterative_process` only. As a result, this function
  does not assert that `before_aggregate` has the expected structure, the caller
  is expected to perform these checks before calling this function.

  Args:
    before_aggregate: The first result of splitting `after_broadcast` on
      aggregate intrinsics.
    grappler_config: An instance of `tf.compat.v1.ConfigProto` to configure
      Grappler graph optimization.

  Returns:
    `work` as specified by `forms.MapReduceForm`, an instance of
    `building_blocks.CompiledComputation`.

  Raises:
    transformations.MapReduceFormCompilationError: If we extract an AST of the
      wrong type.
  """
  c3_elements_in_before_aggregate_parameter = [(0, 1), (1,)]
  c3_to_before_aggregate_computation = _as_function_of_some_federated_subparameters(
      before_aggregate, c3_elements_in_before_aggregate_parameter)
  c4_index_in_before_aggregate_result = [[0, 0], [1, 0]]
  c3_to_unzipped_c4_computation = transformations.select_output_from_lambda(
      c3_to_before_aggregate_computation, c4_index_in_before_aggregate_result)
  c3_to_c4_computation = building_blocks.Lambda(
      c3_to_unzipped_c4_computation.parameter_name,
      c3_to_unzipped_c4_computation.parameter_type,
      building_block_factory.create_federated_zip(
          c3_to_unzipped_c4_computation.result))
  return transformations.consolidate_and_extract_local_processing(
      c3_to_c4_computation, grappler_config)


def _extract_federated_aggregate_functions(before_aggregate, grappler_config):
  """Extracts federated aggregate functions from `before_aggregate`.

  This function is intended to be used by
  `get_map_reduce_form_for_iterative_process` only. As a result, this function
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
    `building_blocks.CompiledComputation`.

  Raises:
    transformations.MapReduceFormCompilationError: If we extract an ASTs of the
      wrong type.
  """
  federated_aggregate = transformations.select_output_from_lambda(
      before_aggregate, 'federated_aggregate_param')
  zero_index_in_federated_aggregate_result = 1
  zero_tff = transformations.select_output_from_lambda(
      federated_aggregate, zero_index_in_federated_aggregate_result).result
  zero_tff = building_blocks.Lambda(None, None, zero_tff)
  accumulate_index_in_federated_aggregate_result = 2
  accumulate_tff = transformations.select_output_from_lambda(
      federated_aggregate,
      accumulate_index_in_federated_aggregate_result).result
  merge_index_in_federated_aggregate_result = 3
  merge_tff = transformations.select_output_from_lambda(
      federated_aggregate, merge_index_in_federated_aggregate_result).result
  report_index_in_federated_aggregate_result = 4
  report_tff = transformations.select_output_from_lambda(
      federated_aggregate, report_index_in_federated_aggregate_result).result

  zero = transformations.consolidate_and_extract_local_processing(
      zero_tff, grappler_config)
  accumulate = transformations.consolidate_and_extract_local_processing(
      accumulate_tff, grappler_config)
  merge = transformations.consolidate_and_extract_local_processing(
      merge_tff, grappler_config)
  report = transformations.consolidate_and_extract_local_processing(
      report_tff, grappler_config)
  return zero, accumulate, merge, report


def _extract_federated_secure_sum_bitwidth_functions(before_aggregate,
                                                     grappler_config):
  """Extracts secure sum from `before_aggregate`.

  This function is intended to be used by
  `get_map_reduce_form_for_iterative_process` only. As a result, this function
  does not assert that `before_aggregate` has the expected structure, the
  caller is expected to perform these checks before calling this function.

  Args:
    before_aggregate: The first result of splitting `after_broadcast` on
      aggregate intrinsics.
    grappler_config: An instance of `tf.compat.v1.ConfigProto` to configure
      Grappler graph optimization.

  Returns:
    `bitwidth` as specified by `forms.MapReduceForm`, an instance of
    `building_blocks.CompiledComputation`.

  Raises:
    transformations.MapReduceFormCompilationError: If we extract an AST of the
      wrong type.
  """
  federated_secure_sum_bitwidth_index_in_before_aggregate_result = 1
  federated_secure_sum_bitwidth = transformations.select_output_from_lambda(
      before_aggregate,
      federated_secure_sum_bitwidth_index_in_before_aggregate_result)
  bitwidth_index_in_federated_secure_sum_bitwidth_result = 1
  bitwidth_tff = transformations.select_output_from_lambda(
      federated_secure_sum_bitwidth,
      bitwidth_index_in_federated_secure_sum_bitwidth_result).result
  bitwidth_tff = building_blocks.Lambda(None, None, bitwidth_tff)

  return transformations.consolidate_and_extract_local_processing(
      bitwidth_tff, grappler_config)


def _extract_update(after_aggregate, grappler_config):
  """Extracts `update` from `after_aggregate`.

  This function is intended to be used by
  `get_map_reduce_form_for_iterative_process` only. As a result, this function
  does not assert that `after_aggregate` has the expected structure, the
  caller is expected to perform these checks before calling this function.

  Args:
    after_aggregate: The second result of splitting `after_broadcast` on
      aggregate intrinsics.
    grappler_config: An instance of `tf.compat.v1.ConfigProto` to configure
      Grappler graph optimization.

  Returns:
    `update` as specified by `forms.MapReduceForm`, an instance of
    `building_blocks.CompiledComputation`.

  Raises:
    transformations.MapReduceFormCompilationError: If we extract an AST of the
      wrong type.
  """
  s7_elements_in_after_aggregate_result = [0, 1]
  s7_output_extracted = transformations.select_output_from_lambda(
      after_aggregate, s7_elements_in_after_aggregate_result)
  s7_output_zipped = building_blocks.Lambda(
      s7_output_extracted.parameter_name, s7_output_extracted.parameter_type,
      building_block_factory.create_federated_zip(s7_output_extracted.result))
  # `create_federated_zip` doesn't have unique reference names, but we need
  # them for `as_function_of_some_federated_subparameters`.
  s7_output_zipped, _ = tree_transformations.uniquify_reference_names(
      s7_output_zipped)
  s6_elements_in_after_aggregate_parameter = [(0, 0, 0), (1, 0), (1, 1)]
  s6_to_s7_computation = _as_function_of_some_federated_subparameters(
      s7_output_zipped, s6_elements_in_after_aggregate_parameter)

  # TODO(b/148942011): The transformation
  # `zip_selection_as_argument_to_lower_level_lambda` does not support selecting
  # from nested structures, therefore we need to pack the type signature
  # `<s1, s3, s4>` as `<s1, <s3, s4>>`.
  name_generator = building_block_factory.unique_name_generator(
      s6_to_s7_computation)

  pack_ref_name = next(name_generator)
  pack_ref_type = computation_types.StructType([
      s6_to_s7_computation.parameter_type.member[0],
      computation_types.StructType([
          s6_to_s7_computation.parameter_type.member[1],
          s6_to_s7_computation.parameter_type.member[2],
      ]),
  ])
  pack_ref = building_blocks.Reference(pack_ref_name, pack_ref_type)
  sel_s1 = building_blocks.Selection(pack_ref, index=0)
  sel = building_blocks.Selection(pack_ref, index=1)
  sel_s3 = building_blocks.Selection(sel, index=0)
  sel_s4 = building_blocks.Selection(sel, index=1)
  result = building_blocks.Struct([sel_s1, sel_s3, sel_s4])
  pack_fn = building_blocks.Lambda(pack_ref.name, pack_ref.type_signature,
                                   result)
  ref_name = next(name_generator)
  ref_type = computation_types.FederatedType(pack_ref_type, placements.SERVER)
  ref = building_blocks.Reference(ref_name, ref_type)
  unpacked_args = building_block_factory.create_federated_map_or_apply(
      pack_fn, ref)
  call = building_blocks.Call(s6_to_s7_computation, unpacked_args)
  fn = building_blocks.Lambda(ref.name, ref.type_signature, call)
  return transformations.consolidate_and_extract_local_processing(
      fn, grappler_config)


def _replace_lambda_body_with_call_dominant_form(
    comp: building_blocks.Lambda) -> building_blocks.Lambda:
  """Transforms the body of `comp` to call-dominant form.

  Call-dominant form ensures that all higher-order functions are fully
  resolved, as well that called intrinsics are pulled out into a top-level
  let-binding. This combination of condition ensures first that pattern-matching
  on calls to intrinsics is sufficient to identify communication operators in
  `force_align_and_split_by_intrinsics`, and second that there are no nested
  intrinsics which will cause that function to fail.

  Args:
    comp: `building_blocks.Lambda` the body of which to convert to call-dominant
      form.

  Returns:
    A transformed version of `comp`, whose body is call-dominant.
  """
  lam_result = comp.result
  result_as_call_dominant, _ = compiler_transformations.transform_to_call_dominant(
      lam_result)
  return building_blocks.Lambda(comp.parameter_name, comp.parameter_type,
                                result_as_call_dominant)


def _merge_grappler_config_with_default(
    grappler_config: tf.compat.v1.ConfigProto) -> tf.compat.v1.ConfigProto:
  py_typecheck.check_type(grappler_config, tf.compat.v1.ConfigProto)
  overridden_grappler_config = tf.compat.v1.ConfigProto()
  overridden_grappler_config.CopyFrom(_GRAPPLER_DEFAULT_CONFIG)
  overridden_grappler_config.MergeFrom(grappler_config)
  return overridden_grappler_config


def get_broadcast_form_for_computation(
    comp: computation_base.Computation,
    grappler_config: tf.compat.v1.ConfigProto = _GRAPPLER_DEFAULT_CONFIG
) -> forms.BroadcastForm:
  """Constructs `tff.backends.mapreduce.BroadcastForm` given a computation.

  Args:
    comp: An instance of `tff.Computation` that is compatible with broadcast
      form. Computations are only compatible if they take in a single value
      placed at server, return a single value placed at clients, and do not
      contain any aggregations.
    grappler_config: An instance of `tf.compat.v1.ConfigProto` to configure
      Grappler graph optimization of the Tensorflow graphs backing the resulting
      `tff.backends.mapreduce.BroadcastForm`. These options are combined with a
      set of defaults that aggressively configure Grappler. If
      `grappler_config_proto` has
      `graph_options.rewrite_options.disable_meta_optimizer=True`, Grappler is
      bypassed.

  Returns:
    An instance of `tff.backends.mapreduce.BroadcastForm` equivalent to the
    provided `tff.Computation`.
  """
  py_typecheck.check_type(comp, computation_base.Computation)
  _check_function_signature_compatible_with_broadcast_form(comp.type_signature)
  py_typecheck.check_type(grappler_config, tf.compat.v1.ConfigProto)
  grappler_config = _merge_grappler_config_with_default(grappler_config)

  bb = comp.to_building_block()
  bb, _ = intrinsic_reductions.replace_intrinsics_with_bodies(bb)
  bb = _replace_lambda_body_with_call_dominant_form(bb)

  tree_analysis.check_contains_only_reducible_intrinsics(bb)
  aggregations = tree_analysis.find_aggregations_in_tree(bb)
  if aggregations:
    raise ValueError(
        f'`get_broadcast_form_for_computation` called with computation '
        f'containing {len(aggregations)} aggregations, but broadcast form '
        'does not allow aggregation. Full list of aggregations:\n{aggregations}'
    )

  before_broadcast, after_broadcast = _split_ast_on_broadcast(bb)
  compute_server_context = _extract_compute_server_context(
      before_broadcast, grappler_config)
  client_processing = _extract_client_processing(after_broadcast,
                                                 grappler_config)

  compute_server_context, client_processing = (
      computation_wrapper_instances.building_block_to_computation(bb)
      for bb in (compute_server_context, client_processing))

  comp_param_names = structure.name_list_with_nones(
      comp.type_signature.parameter)
  server_data_label, client_data_label = comp_param_names
  return forms.BroadcastForm(
      compute_server_context,
      client_processing,
      server_data_label=server_data_label,
      client_data_label=client_data_label)


def get_map_reduce_form_for_iterative_process(
    ip: iterative_process.IterativeProcess,
    grappler_config: tf.compat.v1.ConfigProto = _GRAPPLER_DEFAULT_CONFIG
) -> forms.MapReduceForm:
  """Constructs `tff.backends.mapreduce.MapReduceForm` given iterative process.

  Args:
    ip: An instance of `tff.templates.IterativeProcess` that is compatible with
      MapReduce form. Iterative processes are only compatible if `initialize_fn`
      returns a single federated value placed at `SERVER` and `next` takes
      exactly two arguments. The first must be the state value placed at
      `SERVER`. - `next` returns exactly two values.
    grappler_config: An optional instance of `tf.compat.v1.ConfigProto` to
      configure Grappler graph optimization of the TensorFlow graphs backing the
      resulting `tff.backends.mapreduce.MapReduceForm`. These options are
      combined with a set of defaults that aggressively configure Grappler. If
      the input `grappler_config` has
      `graph_options.rewrite_options.disable_meta_optimizer=True`, Grappler is
      bypassed.

  Returns:
    An instance of `tff.backends.mapreduce.MapReduceForm` equivalent to the
    provided `tff.templates.IterativeProcess`.

  Raises:
    TypeError: If the arguments are of the wrong types.
    transformations.MapReduceFormCompilationError: If the compilation
      process fails.
  """
  py_typecheck.check_type(ip, iterative_process.IterativeProcess)
  initialize_bb, next_bb = (
      check_iterative_process_compatible_with_map_reduce_form(ip))
  py_typecheck.check_type(grappler_config, tf.compat.v1.ConfigProto)
  grappler_config = _merge_grappler_config_with_default(grappler_config)

  next_bb, _ = tree_transformations.uniquify_reference_names(next_bb)
  before_broadcast, after_broadcast = _split_ast_on_broadcast(next_bb)
  before_aggregate, after_aggregate = _split_ast_on_aggregate(after_broadcast)

  initialize = transformations.consolidate_and_extract_local_processing(
      initialize_bb, grappler_config)
  prepare = _extract_prepare(before_broadcast, grappler_config)
  work = _extract_work(before_aggregate, grappler_config)
  zero, accumulate, merge, report = _extract_federated_aggregate_functions(
      before_aggregate, grappler_config)
  bitwidth = _extract_federated_secure_sum_bitwidth_functions(
      before_aggregate, grappler_config)
  update = _extract_update(after_aggregate, grappler_config)

  next_parameter_names = structure.name_list_with_nones(
      ip.next.type_signature.parameter)
  server_state_label, client_data_label = next_parameter_names
  comps = (
      computation_wrapper_instances.building_block_to_computation(bb)
      for bb in (initialize, prepare, work, zero, accumulate, merge, report,
                 bitwidth, update))
  return forms.MapReduceForm(
      *comps,
      server_state_label=server_state_label,
      client_data_label=client_data_label)

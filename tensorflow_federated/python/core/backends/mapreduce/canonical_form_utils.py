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
"""Utils for converting to/from the canonical form.

Note: Refer to `get_iterative_process_for_canonical_form()` for the meaning of
variable names used in this module.
"""

import collections
from typing import Callable

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.backends.mapreduce import canonical_form
from tensorflow_federated.python.core.backends.mapreduce import transformations
from tensorflow_federated.python.core.impl import value_transformations
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper_instances
from tensorflow_federated.python.core.templates import iterative_process


def get_iterative_process_for_canonical_form(cf):
  """Creates `tff.templates.IterativeProcess` from a canonical form.

  Args:
    cf: An instance of `tff.backends.mapreduce.CanonicalForm`.

  Returns:
    An instance of `tff.templates.IterativeProcess` that corresponds to `cf`.

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  py_typecheck.check_type(cf, canonical_form.CanonicalForm)

  @computations.federated_computation
  def init_computation():
    return intrinsics.federated_value(cf.initialize(), placements.SERVER)

  @computations.federated_computation(init_computation.type_signature.result,
                                      computation_types.FederatedType(
                                          cf.work.type_signature.parameter[0],
                                          placements.CLIENTS))
  def next_computation(arg):
    """The logic of a single MapReduce processing round."""
    s1 = arg[0]
    c1 = arg[1]
    s2 = intrinsics.federated_map(cf.prepare, s1)
    c2 = intrinsics.federated_broadcast(s2)
    c3 = intrinsics.federated_zip([c1, c2])
    c4 = intrinsics.federated_map(cf.work, c3)
    c5 = c4[0]
    c6 = c4[1]
    s3 = intrinsics.federated_aggregate(c5, cf.zero(), cf.accumulate, cf.merge,
                                        cf.report)
    s4 = intrinsics.federated_secure_sum(c6, cf.bitwidth())
    s5 = intrinsics.federated_zip([s3, s4])
    s6 = intrinsics.federated_zip([s1, s5])
    s7 = intrinsics.federated_map(cf.update, s6)
    s8 = s7[0]
    s9 = s7[1]
    return s8, s9

  return iterative_process.IterativeProcess(init_computation, next_computation)


def _check_len(
    target,
    length,
    err_fn: Callable[[str],
                     Exception] = transformations.CanonicalFormCompilationError,
):
  py_typecheck.check_type(length, int)
  if len(target) != length:
    raise err_fn('Expected length of {}, found {}.'.format(length, len(target)))


def _check_placement(
    target,
    placement: placement_literals.PlacementLiteral,
    err_fn: Callable[[str],
                     Exception] = transformations.CanonicalFormCompilationError,
):
  py_typecheck.check_type(target, computation_types.FederatedType)
  py_typecheck.check_type(placement, placement_literals.PlacementLiteral)
  if target.placement != placement:
    raise err_fn(
        'Expected value with placement {}, found value of type {}.'.format(
            placement, target))


def _check_type_equal(
    actual,
    expected,
    err_fn: Callable[[str],
                     Exception] = transformations.CanonicalFormCompilationError,
):
  py_typecheck.check_type(actual, computation_types.Type)
  py_typecheck.check_type(expected, computation_types.Type)
  if not actual.is_equivalent_to(expected):
    raise err_fn('Expected type of {}, found {}.'.format(expected, actual))


def _check_type(
    target,
    type_spec,
    err_fn: Callable[[str],
                     Exception] = transformations.CanonicalFormCompilationError,
):
  py_typecheck.check_type(type_spec, type)
  if not isinstance(target, type_spec):
    raise err_fn('Expected type of {}, found {}.'.format(
        type_spec, type(target)))


def _check_type_is_no_arg_fn(
    target,
    err_fn: Callable[[str],
                     Exception] = transformations.CanonicalFormCompilationError,
):
  _check_type(target, computation_types.FunctionType, err_fn)
  if target.parameter is not None:
    raise err_fn(('Expected function to take no argument, but found '
                  'parameter of type {}.').format(target.parameter))


def _check_iterative_process_compatible_with_canonical_form(
    initialize_tree, next_tree):
  """Tests compatibility with `tff.backends.mapreduce.CanonicalForm`.

  Args:
    initialize_tree: An instance of `building_blocks.ComputationBuildingBlock`
      representing the `initalize` component of an
      `tff.templates.IterativeProcess`.
    next_tree: An instance of `building_blocks.ComputationBuildingBlock` that
      representing `next` component of an `tff.templates.IterativeProcess`.

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  py_typecheck.check_type(initialize_tree,
                          building_blocks.ComputationBuildingBlock)
  init_tree_ty = initialize_tree.type_signature
  _check_type_is_no_arg_fn(init_tree_ty, TypeError)
  _check_type(init_tree_ty.result, computation_types.FederatedType, TypeError)
  _check_placement(init_tree_ty.result, placements.SERVER, TypeError)
  py_typecheck.check_type(next_tree, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(next_tree.type_signature,
                          computation_types.FunctionType)
  py_typecheck.check_type(next_tree.type_signature.parameter,
                          computation_types.StructType)
  py_typecheck.check_len(next_tree.type_signature.parameter, 2)
  py_typecheck.check_type(next_tree.type_signature.result,
                          computation_types.StructType)
  py_typecheck.check_len(next_tree.type_signature.parameter, 2)
  next_result_len = len(next_tree.type_signature.result)
  if next_result_len != 2:
    raise TypeError('Expected length of 2, found {}.'.format(next_result_len))


def _create_before_and_after_broadcast_for_no_broadcast(tree):
  r"""Creates a before and after broadcast computations for the given `tree`.

  This function returns the two ASTs:

  Lambda
  |
  Tuple
  |
  []

       Lambda(x)
       |
       Call
      /    \
  Comp      Sel(0)
           /
     Ref(x)

  The first AST is an empty structure that has a type signature satisfying the
  requirements of before broadcast.

  In the second AST, `Comp` is `tree`; `Lambda` has a type signature satisfying
  the requirements of after broadcast; and the argument passed to `Comp` is a
  selection from the parameter of `Lambda` which intentionally drops `c2` on the
  floor.

  This function is intended to be used by
  `get_canonical_form_for_iterative_process` to create before and after
  broadcast computations for the given `tree` when there is no
  `intrinsic_defs.FEDERATED_BROADCAST` in `tree`. As a result, this function
  does not assert that there is no `intrinsic_defs.FEDERATED_BROADCAST` in
  `tree` and it does not assert that `tree` has the expected structure, the
  caller is expected to perform these checks before calling this function.

  Args:
    tree: An instance of `building_blocks.ComputationBuildingBlock`.

  Returns:
    A pair of the form `(before, after)`, where each of `before` and `after`
    is a `tff_framework.ComputationBuildingBlock` that represents a part of the
    result as specified by
    `transformations.force_align_and_split_by_intrinsics`.
  """
  name_generator = building_block_factory.unique_name_generator(tree)

  parameter_name = next(name_generator)
  empty_tuple = building_blocks.Struct([])
  value = building_block_factory.create_federated_value(empty_tuple,
                                                        placements.SERVER)
  before_broadcast = building_blocks.Lambda(parameter_name,
                                            tree.type_signature.parameter,
                                            value)

  parameter_name = next(name_generator)
  type_signature = computation_types.FederatedType(
      before_broadcast.type_signature.result.member, placements.CLIENTS)
  parameter_type = computation_types.StructType(
      [tree.type_signature.parameter, type_signature])
  ref = building_blocks.Reference(parameter_name, parameter_type)
  arg = building_blocks.Selection(ref, index=0)
  call = building_blocks.Call(tree, arg)
  after_broadcast = building_blocks.Lambda(ref.name, ref.type_signature, call)

  return before_broadcast, after_broadcast


def _create_before_and_after_aggregate_for_no_federated_aggregate(tree):
  r"""Creates a before and after aggregate computations for the given `tree`.

  This function returns the two ASTs:

  Lambda
  |
  Tuple
  |
  [Tuple, Comp]
   |
   [Tuple, [], Lambda, Lambda, Lambda]
    |          |       |       |
    []         []      []      []

       Lambda(x)
       |
       Call
      /    \
  Comp      Tuple
            |
            [Sel(0),      Sel(1)]
            /            /
         Ref(x)    Sel(1)
                  /
            Ref(x)

  In the first AST, the second element returned by `Lambda`, `Comp`, is the
  result of the before aggregate returned by force aligning and splitting `tree`
  by `intrinsic_defs.FEDERATED_SECURE_SUM.uri` and the first element returned by
  `Lambda` is an empty structure that represents the argument to the federated
  aggregate intrinsic. Therefore, the first AST has a type signature satisfying
  the requirements of before aggregate.

  In the second AST, `Comp` is the after aggregate returned by force aligning
  and splitting `tree` by intrinsic_defs.FEDERATED_SECURE_SUM.uri; `Lambda` has
  a type signature satisfying the requirements of after aggregate; and the
  argument passed to `Comp` is a selection from the parameter of `Lambda` which
  intentionally drops `s3` on the floor.

  This function is intended to be used by
  `get_canonical_form_for_iterative_process` to create before and after
  broadcast computations for the given `tree` when there is no
  `intrinsic_defs.FEDERATED_AGGREGATE` in `tree`. As a result, this function
  does not assert that there is no `intrinsic_defs.FEDERATED_AGGREGATE` in
  `tree` and it does not assert that `tree` has the expected structure, the
  caller is expected to perform these checks before calling this function.

  Args:
    tree: An instance of `building_blocks.ComputationBuildingBlock`.

  Returns:
    A pair of the form `(before, after)`, where each of `before` and `after`
    is a `tff_framework.ComputationBuildingBlock` that represents a part of the
    result as specified by
    `transformations.force_align_and_split_by_intrinsics`.
  """
  name_generator = building_block_factory.unique_name_generator(tree)

  before_aggregate, after_aggregate = (
      transformations.force_align_and_split_by_intrinsics(
          tree, [intrinsic_defs.FEDERATED_SECURE_SUM.uri]))

  def _create_empty_function(type_elements):
    ref_name = next(name_generator)
    ref_type = computation_types.StructType(type_elements)
    ref = building_blocks.Reference(ref_name, ref_type)
    empty_tuple = building_blocks.Struct([])
    return building_blocks.Lambda(ref.name, ref.type_signature, empty_tuple)

  empty_tuple = building_blocks.Struct([])
  value = building_block_factory.create_federated_value(empty_tuple,
                                                        placements.CLIENTS)
  zero = empty_tuple
  accumulate = _create_empty_function([[], []])
  merge = _create_empty_function([[], []])
  report = _create_empty_function([])
  args = building_blocks.Struct([value, zero, accumulate, merge, report])
  result = building_blocks.Struct([args, before_aggregate.result])
  before_aggregate = building_blocks.Lambda(before_aggregate.parameter_name,
                                            before_aggregate.parameter_type,
                                            result)

  ref_name = next(name_generator)
  s3_type = computation_types.FederatedType([], placements.SERVER)
  ref_type = computation_types.StructType([
      after_aggregate.parameter_type[0],
      computation_types.StructType([s3_type,
                                    after_aggregate.parameter_type[1]]),
  ])
  ref = building_blocks.Reference(ref_name, ref_type)
  sel_arg = building_blocks.Selection(ref, index=0)
  sel = building_blocks.Selection(ref, index=1)
  sel_s4 = building_blocks.Selection(sel, index=1)
  arg = building_blocks.Struct([sel_arg, sel_s4])
  call = building_blocks.Call(after_aggregate, arg)
  after_aggregate = building_blocks.Lambda(ref.name, ref.type_signature, call)

  return before_aggregate, after_aggregate


def _create_before_and_after_aggregate_for_no_federated_secure_sum(tree):
  r"""Creates a before and after aggregate computations for the given `tree`.

  Lambda
  |
  Tuple
  |
  [Comp, Tuple]
         |
         [Tuple, []]
          |
          []

       Lambda(x)
       |
       Call
      /    \
  Comp      Tuple
            |
            [Sel(0),      Sel(0)]
            /            /
         Ref(x)    Sel(1)
                  /
            Ref(x)

  In the first AST, the first element returned by `Lambda`, `Comp`, is the
  result of the before aggregate returned by force aligning and splitting `tree`
  by `intrinsic_defs.FEDERATED_AGGREGATE.uri` and the second element returned by
  `Lambda` is an empty structure that represents the argument to the secure sum
  intrinsic. Therefore, the first AST has a type signature satisfying the
  requirements of before aggregate.

  In the second AST, `Comp` is the after aggregate returned by force aligning
  and splitting `tree` by intrinsic_defs.FEDERATED_AGGREGATE.uri; `Lambda` has a
  type signature satisfying the requirements of after aggregate; and the
  argument passed to `Comp` is a selection from the parameter of `Lambda` which
  intentionally drops `s4` on the floor.

  This function is intended to be used by
  `get_canonical_form_for_iterative_process` to create before and after
  broadcast computations for the given `tree` when there is no
  `intrinsic_defs.FEDERATED_SECURE_SUM` in `tree`. As a result, this function
  does not assert that there is no `intrinsic_defs.FEDERATED_SECURE_SUM` in
  `tree` and it does not assert that `tree` has the expected structure, the
  caller is expected to perform these checks before calling this function.

  Args:
    tree: An instance of `building_blocks.ComputationBuildingBlock`.

  Returns:
    A pair of the form `(before, after)`, where each of `before` and `after`
    is a `tff_framework.ComputationBuildingBlock` that represents a part of the
    result as specified by
    `transformations.force_align_and_split_by_intrinsics`.
  """
  name_generator = building_block_factory.unique_name_generator(tree)

  before_aggregate, after_aggregate = (
      transformations.force_align_and_split_by_intrinsics(
          tree, [intrinsic_defs.FEDERATED_AGGREGATE.uri]))

  empty_tuple = building_blocks.Struct([])
  value = building_block_factory.create_federated_value(empty_tuple,
                                                        placements.CLIENTS)
  bitwidth = empty_tuple
  args = building_blocks.Struct([value, bitwidth])
  result = building_blocks.Struct([before_aggregate.result, args])
  before_aggregate = building_blocks.Lambda(before_aggregate.parameter_name,
                                            before_aggregate.parameter_type,
                                            result)

  ref_name = next(name_generator)
  s4_type = computation_types.FederatedType([], placements.SERVER)
  ref_type = computation_types.StructType([
      after_aggregate.parameter_type[0],
      computation_types.StructType([
          after_aggregate.parameter_type[1],
          s4_type,
      ]),
  ])
  ref = building_blocks.Reference(ref_name, ref_type)
  sel_arg = building_blocks.Selection(ref, index=0)
  sel = building_blocks.Selection(ref, index=1)
  sel_s3 = building_blocks.Selection(sel, index=0)
  arg = building_blocks.Struct([sel_arg, sel_s3])
  call = building_blocks.Call(after_aggregate, arg)
  after_aggregate = building_blocks.Lambda(ref.name, ref.type_signature, call)

  return before_aggregate, after_aggregate


def _extract_prepare(before_broadcast):
  """extracts `prepare` from `before_broadcast`.

  This function is intended to be used by
  `get_canonical_form_for_iterative_process` only. As a result, this function
  does not assert that `before_broadcast` has the expected structure, the
  caller is expected to perform these checks before calling this function.

  Args:
    before_broadcast: The first result of splitting `next_comp` on
      `intrinsic_defs.FEDERATED_BROADCAST`.

  Returns:
    `prepare` as specified by `canonical_form.CanonicalForm`, an instance of
    `building_blocks.CompiledComputation`.

  Raises:
    transformations.CanonicalFormCompilationError: If we extract an AST of the
      wrong type.
  """
  s1_index_in_before_broadcast = 0
  s1_to_s2_computation = (
      transformations.bind_single_selection_as_argument_to_lower_level_lambda(
          before_broadcast, s1_index_in_before_broadcast)).result.function
  return transformations.consolidate_and_extract_local_processing(
      s1_to_s2_computation)


def _extract_work(before_aggregate):
  """Extracts `work` from `before_aggregate`.

  This function is intended to be used by
  `get_canonical_form_for_iterative_process` only. As a result, this function
  does not assert that `before_aggregate` has the expected structure, the caller
  is expected to perform these checks before calling this function.

  Args:
    before_aggregate: The first result of splitting `after_broadcast` on
      aggregate intrinsics.

  Returns:
    `work` as specified by `canonical_form.CanonicalForm`, an instance of
    `building_blocks.CompiledComputation`.

  Raises:
    transformations.CanonicalFormCompilationError: If we extract an AST of the
      wrong type.
  """
  c3_elements_in_before_aggregate_parameter = [[0, 1], [1]]
  c3_to_before_aggregate_computation = (
      transformations.zip_selection_as_argument_to_lower_level_lambda(
          before_aggregate,
          c3_elements_in_before_aggregate_parameter).result.function)
  c4_index_in_before_aggregate_result = [[0, 0], [1, 0]]
  c3_to_unzipped_c4_computation = transformations.select_output_from_lambda(
      c3_to_before_aggregate_computation, c4_index_in_before_aggregate_result)
  c3_to_c4_computation = building_blocks.Lambda(
      c3_to_unzipped_c4_computation.parameter_name,
      c3_to_unzipped_c4_computation.parameter_type,
      building_block_factory.create_federated_zip(
          c3_to_unzipped_c4_computation.result))

  return transformations.consolidate_and_extract_local_processing(
      c3_to_c4_computation)


def _extract_federated_aggregate_functions(before_aggregate):
  """Extracts federated aggregate functions from `before_aggregate`.

  This function is intended to be used by
  `get_canonical_form_for_iterative_process` only. As a result, this function
  does not assert that `before_aggregate` has the expected structure, the
  caller is expected to perform these checks before calling this function.

  Args:
    before_aggregate: The first result of splitting `after_broadcast` on
      aggregate intrinsics.

  Returns:
    `zero`, `accumulate`, `merge` and `report` as specified by
    `canonical_form.CanonicalForm`. All are instances of
    `building_blocks.CompiledComputation`.

  Raises:
    transformations.CanonicalFormCompilationError: If we extract an ASTs of the
      wrong type.
  """
  federated_aggregate_index_in_before_aggregate_result = 0
  federated_aggregate = transformations.select_output_from_lambda(
      before_aggregate, federated_aggregate_index_in_before_aggregate_result)
  zero_index_in_federated_aggregate_result = 1
  zero_tff = transformations.select_output_from_lambda(
      federated_aggregate, zero_index_in_federated_aggregate_result).result
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

  zero = transformations.consolidate_and_extract_local_processing(zero_tff)
  accumulate = transformations.consolidate_and_extract_local_processing(
      accumulate_tff)
  merge = transformations.consolidate_and_extract_local_processing(merge_tff)
  report = transformations.consolidate_and_extract_local_processing(report_tff)
  return zero, accumulate, merge, report


def _extract_federated_secure_sum_functions(before_aggregate):
  """Extracts secure sum from `before_aggregate`.

  This function is intended to be used by
  `get_canonical_form_for_iterative_process` only. As a result, this function
  does not assert that `before_aggregate` has the expected structure, the
  caller is expected to perform these checks before calling this function.

  Args:
    before_aggregate: The first result of splitting `after_broadcast` on
      aggregate intrinsics.

  Returns:
    `bitwidth` as specified by `canonical_form.CanonicalForm`, an instance of
    `building_blocks.CompiledComputation`.

  Raises:
    transformations.CanonicalFormCompilationError: If we extract an AST of the
      wrong type.
  """
  federated_secure_sum_index_in_before_aggregate_result = 1
  federated_secure_sum = transformations.select_output_from_lambda(
      before_aggregate, federated_secure_sum_index_in_before_aggregate_result)
  bitwidth_index_in_federated_secure_sum_result = 1
  bitwidth_tff = transformations.select_output_from_lambda(
      federated_secure_sum,
      bitwidth_index_in_federated_secure_sum_result).result

  return transformations.consolidate_and_extract_local_processing(bitwidth_tff)


def _extract_update(after_aggregate):
  """Extracts `update` from `after_aggregate`.

  This function is intended to be used by
  `get_canonical_form_for_iterative_process` only. As a result, this function
  does not assert that `after_aggregate` has the expected structure, the
  caller is expected to perform these checks before calling this function.

  Args:
    after_aggregate: The second result of splitting `after_broadcast` on
      aggregate intrinsics.

  Returns:
    `update` as specified by `canonical_form.CanonicalForm`, an instance of
    `building_blocks.CompiledComputation`.

  Raises:
    transformations.CanonicalFormCompilationError: If we extract an AST of the
      wrong type.
  """
  s7_elements_in_after_aggregate_result = [0, 1]
  s7_output_extracted = transformations.select_output_from_lambda(
      after_aggregate, s7_elements_in_after_aggregate_result)
  s7_output_zipped = building_blocks.Lambda(
      s7_output_extracted.parameter_name, s7_output_extracted.parameter_type,
      building_block_factory.create_federated_zip(s7_output_extracted.result))
  s6_elements_in_after_aggregate_parameter = [[0, 0, 0], [1, 0], [1, 1]]
  s6_to_s7_computation = (
      transformations.zip_selection_as_argument_to_lower_level_lambda(
          s7_output_zipped,
          s6_elements_in_after_aggregate_parameter).result.function)

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
  return transformations.consolidate_and_extract_local_processing(fn)


def _get_type_info(initialize_tree, before_broadcast, after_broadcast,
                   before_aggregate, after_aggregate):
  """Returns type information for an `tff.templates.IterativeProcess`.

  This function is intended to be used by
  `get_canonical_form_for_iterative_process` to create the expected type
  signatures when compiling a given `tff.templates.IterativeProcess` into a
  `tff.backends.mapreduce.CanonicalForm` and returns a `collections.OrderedDict`
  whose keys and order match the explicit and intermediate componets of
  `tff.backends.mapreduce.CanonicalForm` defined here:

  ```
  s1 = arg[0]
  c1 = arg[1]
  s2 = intrinsics.federated_map(cf.prepare, s1)
  c2 = intrinsics.federated_broadcast(s2)
  c3 = intrinsics.federated_zip([c1, c2])
  c4 = intrinsics.federated_map(cf.work, c3)
  c5 = c4[0]
  c6 = c4[1]
  s3 = intrinsics.federated_aggregate(c5,
                                      cf.zero(),
                                      cf.accumulate,
                                      cf.merge,
                                      cf.report)
  s4 = intrinsics.federated_secure_sum(c6, cf.bitwidth())
  s5 = intrinsics.federated_zip([s3, s4])
  s6 = intrinsics.federated_zip([s1, s5])
  s7 = intrinsics.federated_map(cf.update, s6)
  s8 = s7[0]
  s9 = s7[1]
  ```

  Note that the type signatures for the `initalize` and `next` components of an
  `tff.templates.IterativeProcess` are:

  initalize:  `( -> s1)`
  next:       `(<s1,c1> -> <s8,s9>)`

  However, the `next` component of an `tff.templates.IterativeProcess` has been
  split into a before and after broadcast and a before and after aggregate with
  the given semantics:

  ```
  (arg -> after(<arg, intrinsic(before(arg))>))
  ```

  as a result, the type signatures for the components split from the `next`
  component of an `tff.templates.IterativeProcess` are:

  before_broadcast:  `(<s1,c1> -> s2)`
  after_broadcast:   `(<<s1,c1>,c2> -> <s8,s9>)`
  before_aggregate:  `(<<s1,c1>,c2> -> <<c5,zero,accumulate,merge,report>,c6>)`
  after_aggregate:   `(<<<s1,c1>,c2>,<s3,s4>> -> <s8,s9>)`

  Args:
    initialize_tree: An instance of `building_blocks.ComputationBuildingBlock`
      representing the `initalize` component of an
      `tff.templates.IterativeProcess`.
    before_broadcast: The first result of splitting `next` component of an
      `tff.templates.IterativeProcess` on broadcast.
    after_broadcast: The second result of splitting `next` component of an
      `tff.templates.IterativeProcess` on broadcast.
    before_aggregate: The first result of splitting `next` component of an
      `tff.templates.IterativeProcess` on aggregate.
    after_aggregate: The second result of splitting `next` component of an
      `tff.templates.IterativeProcess` on aggregate.

  Raises:
    transformations.CanonicalFormCompilationError: If the arguments are of the
      wrong types.
  """

  # The type signature of `initalize` is: `( -> s1)`.
  init_tree_ty = initialize_tree.type_signature
  _check_type_is_no_arg_fn(init_tree_ty)
  _check_type(init_tree_ty.result, computation_types.FederatedType)
  _check_placement(init_tree_ty.result, placements.SERVER)
  # The named components of canonical form have no placement, so we must
  # remove the placement on the return type of initialize_tree
  initialize_type = computation_types.FunctionType(
      initialize_tree.type_signature.parameter,
      initialize_tree.type_signature.result.member)

  # The type signature of `before_broadcast` is: `(<s1,c1> -> s2)`.
  _check_type(before_broadcast.type_signature, computation_types.FunctionType)
  _check_type(before_broadcast.type_signature.parameter,
              computation_types.StructType)
  _check_len(before_broadcast.type_signature.parameter, 2)
  s1_type = before_broadcast.type_signature.parameter[0]
  _check_type(s1_type, computation_types.FederatedType)
  _check_placement(s1_type, placements.SERVER)
  c1_type = before_broadcast.type_signature.parameter[1]
  _check_type(c1_type, computation_types.FederatedType)
  _check_placement(c1_type, placements.CLIENTS)
  s2_type = before_broadcast.type_signature.result
  _check_type(s2_type, computation_types.FederatedType)
  _check_placement(s2_type, placements.SERVER)

  prepare_type = computation_types.FunctionType(s1_type.member, s2_type.member)

  # The type signature of `after_broadcast` is: `(<<s1,c1>,c2> -> <s8,s9>)'.
  _check_type(after_broadcast.type_signature, computation_types.FunctionType)
  _check_type(after_broadcast.type_signature.parameter,
              computation_types.StructType)
  _check_len(after_broadcast.type_signature.parameter, 2)
  _check_type(after_broadcast.type_signature.parameter[0],
              computation_types.StructType)
  _check_len(after_broadcast.type_signature.parameter[0], 2)
  _check_type_equal(after_broadcast.type_signature.parameter[0][0], s1_type)
  _check_type_equal(after_broadcast.type_signature.parameter[0][1], c1_type)
  c2_type = after_broadcast.type_signature.parameter[1]
  _check_type(c2_type, computation_types.FederatedType)
  _check_placement(c2_type, placements.CLIENTS)
  _check_type(after_broadcast.type_signature.result,
              computation_types.StructType)
  _check_len(after_broadcast.type_signature.result, 2)
  s8_type = after_broadcast.type_signature.result[0]
  _check_type(s8_type, computation_types.FederatedType)
  _check_placement(s8_type, placements.SERVER)
  s9_type = after_broadcast.type_signature.result[1]
  _check_type(s9_type, computation_types.FederatedType)
  _check_placement(s9_type, placements.SERVER)

  # The type signature of `before_aggregate` is:
  # `(<<s1,c1>,c2> -> <<c5,zero,accumulate,merge,report>,<c6,bitwidth>>)`.
  _check_type(before_aggregate.type_signature, computation_types.FunctionType)
  _check_type(before_aggregate.type_signature.parameter,
              computation_types.StructType)
  _check_len(before_aggregate.type_signature.parameter, 2)
  _check_type(before_aggregate.type_signature.parameter[0],
              computation_types.StructType)
  _check_len(before_aggregate.type_signature.parameter[0], 2)
  _check_type_equal(before_aggregate.type_signature.parameter[0][0], s1_type)
  _check_type_equal(before_aggregate.type_signature.parameter[0][1], c1_type)
  _check_type_equal(before_aggregate.type_signature.parameter[1], c2_type)
  _check_type(before_aggregate.type_signature.result,
              computation_types.StructType)
  _check_len(before_aggregate.type_signature.result, 2)
  _check_len(before_aggregate.type_signature.result[0], 5)
  c5_type = before_aggregate.type_signature.result[0][0]
  _check_type(c5_type, computation_types.FederatedType)
  _check_placement(c5_type, placements.CLIENTS)
  zero_type = computation_types.FunctionType(
      None, before_aggregate.type_signature.result[0][1])
  type_analysis.check_tensorflow_compatible_type(zero_type.result)
  accumulate_type = before_aggregate.type_signature.result[0][2]
  _check_type(accumulate_type, computation_types.FunctionType)
  merge_type = before_aggregate.type_signature.result[0][3]
  _check_type(merge_type, computation_types.FunctionType)
  report_type = before_aggregate.type_signature.result[0][4]
  _check_type(report_type, computation_types.FunctionType)
  _check_type(before_aggregate.type_signature.result[1],
              computation_types.StructType)
  _check_len(before_aggregate.type_signature.result[1], 2)
  c6_type = before_aggregate.type_signature.result[1][0]
  _check_type(c6_type, computation_types.FederatedType)
  _check_placement(c6_type, placements.CLIENTS)
  bitwidth_type = computation_types.FunctionType(
      None, before_aggregate.type_signature.result[1][1])
  type_analysis.check_tensorflow_compatible_type(bitwidth_type.result)

  c3_type = computation_types.FederatedType([c1_type.member, c2_type.member],
                                            placements.CLIENTS)
  c4_type = computation_types.FederatedType([c5_type.member, c6_type.member],
                                            placements.CLIENTS)

  # The type signature of `after_aggregate` is:
  # `(<<<s1,c1>,c2>,<s3,s4>> -> <s8,s9>)'.
  _check_type(after_aggregate.type_signature, computation_types.FunctionType)
  _check_type(after_aggregate.type_signature.parameter,
              computation_types.StructType)
  _check_len(after_aggregate.type_signature.parameter, 2)
  _check_type(after_aggregate.type_signature.parameter[0],
              computation_types.StructType)
  _check_len(after_aggregate.type_signature.parameter[0], 2)
  _check_type(after_aggregate.type_signature.parameter[0][0],
              computation_types.StructType)
  _check_len(after_aggregate.type_signature.parameter[0][0], 2)
  _check_type_equal(after_aggregate.type_signature.parameter[0][0][0], s1_type)
  _check_type_equal(after_aggregate.type_signature.parameter[0][0][1], c1_type)
  _check_type_equal(after_aggregate.type_signature.parameter[0][1], c2_type)
  _check_len(after_aggregate.type_signature.parameter[1], 2)
  s3_type = after_aggregate.type_signature.parameter[1][0]
  _check_type(s3_type, computation_types.FederatedType)
  _check_placement(s3_type, placements.SERVER)
  s4_type = after_aggregate.type_signature.parameter[1][1]
  _check_type(s4_type, computation_types.FederatedType)
  _check_placement(s4_type, placements.SERVER)
  _check_len(after_aggregate.type_signature.result, 2)
  _check_type_equal(after_aggregate.type_signature.result[0], s8_type)
  _check_type_equal(after_aggregate.type_signature.result[1], s9_type)

  work_type = computation_types.FunctionType(c3_type.member, c4_type.member)

  s5_type = computation_types.FederatedType([s3_type.member, s4_type.member],
                                            placements.SERVER)
  s6_type = computation_types.FederatedType([s1_type.member, s5_type.member],
                                            placements.SERVER)
  s7_type = computation_types.FederatedType([s8_type.member, s9_type.member],
                                            placements.SERVER)
  update_type = computation_types.FunctionType(s6_type.member, s7_type.member)

  return collections.OrderedDict(
      initialize_type=initialize_type,
      s1_type=s1_type,
      c1_type=c1_type,
      prepare_type=prepare_type,
      s2_type=s2_type,
      c2_type=c2_type,
      c3_type=c3_type,
      work_type=work_type,
      c4_type=c4_type,
      c5_type=c5_type,
      c6_type=c6_type,
      zero_type=zero_type,
      accumulate_type=accumulate_type,
      merge_type=merge_type,
      report_type=report_type,
      s3_type=s3_type,
      bitwidth_type=bitwidth_type,
      s4_type=s4_type,
      s5_type=s5_type,
      s6_type=s6_type,
      update_type=update_type,
      s7_type=s7_type,
      s8_type=s8_type,
      s9_type=s9_type,
  )


def _replace_intrinsics_with_bodies(comp):
  """Replaces intrinsics with their bodies as defined in `intrinsic_bodies.py`.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` in which we
      wish to replace all intrinsics with their bodies.

  Returns:
    An instance of `building_blocks.ComputationBuildingBlock` with
    all intrinsics defined in `intrinsic_bodies.py` replaced with their bodies.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  context_stack = context_stack_impl.context_stack
  comp, _ = value_transformations.replace_intrinsics_with_bodies(
      comp, context_stack)
  return comp


def get_canonical_form_for_iterative_process(ip):
  """Constructs `tff.backends.mapreduce.CanonicalForm` given iterative process.

  This function transforms computations from the input `ip` into
  an instance of `tff.backends.mapreduce.CanonicalForm`.

  Args:
    ip: An instance of `tff.templates.IterativeProcess`.

  Returns:
    An instance of `tff.backends.mapreduce.CanonicalForm` equivalent to this
    process.

  Raises:
    TypeError: If the arguments are of the wrong types.
    transformations.CanonicalFormCompilationError: If the compilation
      process fails.
  """
  py_typecheck.check_type(ip, iterative_process.IterativeProcess)

  initialize_comp = building_blocks.ComputationBuildingBlock.from_proto(
      ip.initialize._computation_proto)  # pylint: disable=protected-access
  next_comp = building_blocks.ComputationBuildingBlock.from_proto(
      ip.next._computation_proto)  # pylint: disable=protected-access
  _check_iterative_process_compatible_with_canonical_form(
      initialize_comp, next_comp)

  initialize_comp = _replace_intrinsics_with_bodies(initialize_comp)
  next_comp = _replace_intrinsics_with_bodies(next_comp)
  tree_analysis.check_contains_only_reducible_intrinsics(initialize_comp)
  tree_analysis.check_contains_only_reducible_intrinsics(next_comp)
  tree_analysis.check_broadcast_not_dependent_on_aggregate(next_comp)

  if tree_analysis.contains_called_intrinsic(
      next_comp, intrinsic_defs.FEDERATED_BROADCAST.uri):
    before_broadcast, after_broadcast = (
        transformations.force_align_and_split_by_intrinsics(
            next_comp, [intrinsic_defs.FEDERATED_BROADCAST.uri]))
  else:
    before_broadcast, after_broadcast = (
        _create_before_and_after_broadcast_for_no_broadcast(next_comp))

  contains_federated_aggregate = tree_analysis.contains_called_intrinsic(
      next_comp, intrinsic_defs.FEDERATED_AGGREGATE.uri)
  contains_federated_secure_sum = tree_analysis.contains_called_intrinsic(
      next_comp, intrinsic_defs.FEDERATED_SECURE_SUM.uri)
  if not (contains_federated_aggregate or contains_federated_secure_sum):
    raise ValueError(
        'Expected an `tff.templates.IterativeProcess` containing at least one '
        '`federated_aggregate` or `federated_secure_sum`, found none.')

  if contains_federated_aggregate and contains_federated_secure_sum:
    before_aggregate, after_aggregate = (
        transformations.force_align_and_split_by_intrinsics(
            after_broadcast, [
                intrinsic_defs.FEDERATED_AGGREGATE.uri,
                intrinsic_defs.FEDERATED_SECURE_SUM.uri,
            ]))
  elif contains_federated_secure_sum:
    assert not contains_federated_aggregate
    before_aggregate, after_aggregate = (
        _create_before_and_after_aggregate_for_no_federated_aggregate(
            after_broadcast))
  else:
    assert contains_federated_aggregate and not contains_federated_secure_sum
    before_aggregate, after_aggregate = (
        _create_before_and_after_aggregate_for_no_federated_secure_sum(
            after_broadcast))

  type_info = _get_type_info(initialize_comp, before_broadcast, after_broadcast,
                             before_aggregate, after_aggregate)

  initialize = transformations.consolidate_and_extract_local_processing(
      initialize_comp)
  _check_type_equal(initialize.type_signature, type_info['initialize_type'])

  prepare = _extract_prepare(before_broadcast)
  _check_type_equal(prepare.type_signature, type_info['prepare_type'])

  work = _extract_work(before_aggregate)
  _check_type_equal(work.type_signature, type_info['work_type'])

  zero, accumulate, merge, report = _extract_federated_aggregate_functions(
      before_aggregate)
  _check_type_equal(zero.type_signature, type_info['zero_type'])
  _check_type_equal(accumulate.type_signature, type_info['accumulate_type'])
  _check_type_equal(merge.type_signature, type_info['merge_type'])
  _check_type_equal(report.type_signature, type_info['report_type'])

  bitwidth = _extract_federated_secure_sum_functions(before_aggregate)
  _check_type_equal(bitwidth.type_signature, type_info['bitwidth_type'])

  update = _extract_update(after_aggregate)
  _check_type_equal(update.type_signature, type_info['update_type'])

  return canonical_form.CanonicalForm(
      computation_wrapper_instances.building_block_to_computation(initialize),
      computation_wrapper_instances.building_block_to_computation(prepare),
      computation_wrapper_instances.building_block_to_computation(work),
      computation_wrapper_instances.building_block_to_computation(zero),
      computation_wrapper_instances.building_block_to_computation(accumulate),
      computation_wrapper_instances.building_block_to_computation(merge),
      computation_wrapper_instances.building_block_to_computation(report),
      computation_wrapper_instances.building_block_to_computation(bitwidth),
      computation_wrapper_instances.building_block_to_computation(update))

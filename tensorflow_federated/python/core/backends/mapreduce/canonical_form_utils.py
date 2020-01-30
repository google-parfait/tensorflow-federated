# Lint as: python3
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
"""Utils for converting to/from the canonical form."""

import itertools

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.backends.mapreduce import canonical_form
from tensorflow_federated.python.core.backends.mapreduce import transformations
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl import value_transformations
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper_instances
from tensorflow_federated.python.core.utils import computation_utils


def get_iterative_process_for_canonical_form(cf):
  """Creates `tff.utils.IterativeProcess` from a canonical form.

  Args:
    cf: An instance of `tff.backends.mapreduce.CanonicalForm`.

  Returns:
    An instance of `tff.utils.IterativeProcess` that corresponds to `cf`.

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
    """The logic of a single MapReduce sprocessing round."""
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
    s4 = intrinsics.federated_zip([s1, s3])
    s5 = intrinsics.federated_map(cf.update, s4)
    s6 = s5[0]
    s7 = s5[1]
    return s6, s7, c6

  return computation_utils.IterativeProcess(init_computation, next_computation)


def _check_type_equal(actual, expected, label):
  if actual != expected:
    raise transformations.CanonicalFormCompilationError(
        'Expected \'{}\' to have a type signature of {}, found {}.'.format(
            label, expected, actual))


def pack_initialize_comp_type_signature(type_spec):
  """Packs the initialize type to be used by the remainder of the compiler."""
  if not (isinstance(type_spec, computation_types.FederatedType) and
          type_spec.placement == placements.SERVER):
    raise TypeError(
        'Expected init type spec to be a federated type placed at the server; '
        'instead found {}'.format(type_spec))
  initialize_type = computation_types.FunctionType(None, type_spec.member)
  return {'initialize_type': initialize_type}


def pack_next_comp_type_signature(type_signature, previously_packed_types):
  """Packs types that can be inferred from `next.type_signature` into a dict.

  The `next` portion of a `tff.utils.IterativeProcess` should have type
  signature `<s1, c1> -> <s6, s7, c6>`, where `sn` and `cn` are as defined in
  `canonical_form.py`.

  Args:
    type_signature: The `type_signature` attribute of the `next` portion of the
      `tff.utils.IterativeProcess` from which we are looking to extract an
      instance of `canonical_form.CanonicalForm`.
    previously_packed_types: A `dict` containing the initialize type.

  Returns:
    A `dict` packing the types which can be inferred from `type_signature`.

  Raises:
    TypeError: If `type_signature` is incompatible with being a type signature
    of the `next` computation of a `tff.utils.IterativeProcess`.
  """
  should_raise = False
  if not (isinstance(type_signature, computation_types.FunctionType) and
          isinstance(type_signature.parameter, computation_types.NamedTupleType)
          and len(type_signature.parameter) == 2 and isinstance(
              type_signature.result, computation_types.NamedTupleType) and
          len(type_signature.result) == 3):
    should_raise = True
  if (type_signature.parameter[0].member !=
      previously_packed_types['initialize_type'].result):
    should_raise = True
  for server_placed_type in [
      type_signature.parameter[0], type_signature.result[0],
      type_signature.result[1]
  ]:
    if not (isinstance(server_placed_type, computation_types.FederatedType) and
            server_placed_type.placement == placements.SERVER):
      should_raise = True

  if len(type_signature.result) == 3:
    for client_placed_type in [
        type_signature.parameter[1], type_signature.result[2]
    ]:
      if not (isinstance(client_placed_type, computation_types.FederatedType)
              and client_placed_type.placement == placements.CLIENTS):
        should_raise = True
  if should_raise:
    # TODO(b/121290421): These error messages, and indeed the 'track boolean and
    # raise once' logic of these methods as well, is intended to be provisional
    # and revisited when we've seen the compilation pipeline fail more clearly,
    # or maybe preferably iteratively improved as new failure modes are
    # encountered.
    raise TypeError(
        'Checking the types of `next` has failed. Expected a '
        'type signature of `<s1,c1> -> <s6,s7,c6>` as defined in '
        '`canonical_form.CanonicalForm`, but given '
        '`next` computation has type signature {}'.format(type_signature))

  newly_determined_types = {}
  newly_determined_types['s1_type'] = type_signature.parameter[0]
  newly_determined_types['c1_type'] = type_signature.parameter[1]
  newly_determined_types['s6_type'] = type_signature.result[0]
  newly_determined_types['s7_type'] = type_signature.result[1]
  newly_determined_types['c6_type'] = type_signature.result[2]
  return dict(
      itertools.chain(previously_packed_types.items(),
                      newly_determined_types.items()))


def check_and_pack_before_broadcast_type_signature(type_spec,
                                                   previously_packed_types):
  """Checks types inferred from `before_broadcast` and packs in `previously_packed_types`.

  After splitting the `next` portion of a `tff.utils.IterativeProcess` into
  `before_broadcast` and `after_broadcast`, `before_broadcast` should have
  type signature `<s1, c1> -> s2`. This function validates `c1` and `s1`
  against the existing entries in `previously_packed_types`, then packs `s2`.

  Args:
    type_spec: The `type_signature` attribute of the `before_broadcast` portion
      of the `tff.utils.IterativeProcess` from which we are looking to extract
      an instance of `canonical_form.CanonicalForm`.
    previously_packed_types: Dict containing the information from `next` in the
      iterative process we are parsing.

  Returns:
    A `dict` packing the types which can be inferred from `type_signature`.

  Raises:
    TypeError: If `type_signature` is incompatible with
    `previously_packed_types`.
  """
  should_raise = False
  if not (isinstance(type_spec, computation_types.FunctionType) and
          isinstance(type_spec.parameter, computation_types.NamedTupleType) and
          len(type_spec.parameter) == 2 and
          type_spec.parameter[0] == previously_packed_types['s1_type'] and
          type_spec.parameter[1] == previously_packed_types['c1_type']):
    should_raise = True
  if not (isinstance(type_spec.result, computation_types.FederatedType) and
          type_spec.result.placement == placements.SERVER):
    should_raise = True
  if should_raise:
    # TODO(b/121290421): These error messages, and indeed the 'track boolean and
    # raise once' logic of these methods as well, is intended to be provisional
    # and revisited when we've seen the compilation pipeline fail more clearly,
    # or maybe preferably iteratively improved as new failure modes are
    # encountered.
    raise TypeError('We have encountered an error checking the type signature '
                    'of `before_broadcast`; expected it to have the form '
                    '`<s1,c1> -> s2`, with `s1` matching {} and `c1` matching '
                    '{}, as defined in `connical_form.CanonicalForm`, but '
                    'encountered a type spec {}'.format(
                        previously_packed_types['s1_type'],
                        previously_packed_types['c1_type'], type_spec))
  s2 = type_spec.result
  newly_determined_types = {}
  newly_determined_types['s2_type'] = s2
  newly_determined_types['prepare_type'] = computation_types.FunctionType(
      previously_packed_types['s1_type'].member, s2.member)
  return dict(
      itertools.chain(previously_packed_types.items(),
                      newly_determined_types.items()))


def check_and_pack_before_aggregate_type_signature(type_spec,
                                                   previously_packed_types):
  """Checks types inferred from `before_aggregate` and packs in `previously_packed_types`.

  After splitting the `after_broadcast` portion of a
  `tff.utils.IterativeProcess` into `before_aggregate` and `after_aggregate`,
  `before_aggregate` should have type signature
  `<<s1,c1>,c2> -> <c5,zero,accumulate,merge,report>`. This
  function validates `c1`, `s1` and `c2` against the existing entries in
  `previously_packed_types`, then packs `s5`, `zero`, `accumulate`, `merge` and
  `report`.

  Args:
    type_spec: The `type_signature` attribute of the `before_aggregate` portion
      of the `tff.utils.IterativeProcess` from which we are looking to extract
      an instance of `canonical_form.CanonicalForm`.
    previously_packed_types: Dict containing the information from `next` and
      `before_broadcast` in the iterative process we are parsing.

  Returns:
    A `dict` packing the types which can be inferred from `type_spec`.

  Raises:
    TypeError: If `type_signature` is incompatible with
    `previously_packed_types`.
  """
  should_raise = False
  if not (isinstance(type_spec, computation_types.FunctionType) and
          isinstance(type_spec.parameter, computation_types.NamedTupleType)):
    should_raise = True
  if not (isinstance(type_spec.parameter[0], computation_types.NamedTupleType)
          and len(type_spec.parameter[0]) == 2 and
          type_spec.parameter[0][0] == previously_packed_types['s1_type'] and
          type_spec.parameter[0][1] == previously_packed_types['c1_type']):
    should_raise = True
  if not (
      isinstance(type_spec.parameter[1], computation_types.FederatedType) and
      type_spec.parameter[1].placement == placements.CLIENTS and
      type_spec.parameter[1].member == previously_packed_types['s2_type'].member
  ):
    should_raise = True
  if not (isinstance(type_spec.result, computation_types.NamedTupleType) and
          len(type_spec.result) == 5 and
          isinstance(type_spec.result[0], computation_types.FederatedType) and
          type_spec.result[0].placement == placements.CLIENTS and
          type_utils.is_tensorflow_compatible_type(type_spec.result[1]) and
          type_spec.result[2] == computation_types.FunctionType(
              [type_spec.result[1], type_spec.result[0].member],
              type_spec.result[1]) and
          type_spec.result[3] == computation_types.FunctionType(
              [type_spec.result[1], type_spec.result[1]], type_spec.result[1])
          and type_spec.result[4].parameter == type_spec.result[1] and
          type_utils.is_tensorflow_compatible_type(type_spec.result[4].result)):
    should_raise = True
  if should_raise:
    # TODO(b/121290421): These error messages, and indeed the 'track boolean and
    # raise once' logic of these methods as well, is intended to be provisional
    # and revisited when we've seen the compilation pipeline fail more clearly,
    # or maybe preferably iteratively improved as new failure modes are
    # encountered.
    raise TypeError(
        'Encountered a type error while checking '
        '`before_aggregate`. Expected a type signature of the '
        'form `<<s1,c1>,c2> -> <c5,zero,accumulate,merge,report>`, '
        'where `s1` matches {}, `c1` matches {}, and `c2` matches '
        'the result of broadcasting {}, as defined in '
        '`canonical_form.CanonicalForm`. Found type signature {}.'.format(
            previously_packed_types['s1_type'],
            previously_packed_types['c1_type'],
            previously_packed_types['s2_type'], type_spec))
  newly_determined_types = {}
  c2_type = type_spec.parameter[1]
  newly_determined_types['c2_type'] = c2_type
  c3_type = computation_types.FederatedType(
      [previously_packed_types['c1_type'].member, c2_type.member],
      placements.CLIENTS)
  newly_determined_types['c3_type'] = c3_type
  c5_type = type_spec.result[0]
  zero_type = computation_types.FunctionType(None, type_spec.result[1])
  accumulate_type = type_spec.result[2]
  merge_type = type_spec.result[3]
  report_type = type_spec.result[4]
  newly_determined_types['c5_type'] = c5_type
  newly_determined_types['zero_type'] = zero_type
  newly_determined_types['accumulate_type'] = accumulate_type
  newly_determined_types['merge_type'] = merge_type
  newly_determined_types['report_type'] = report_type
  newly_determined_types['s3_type'] = computation_types.FederatedType(
      report_type.result, placements.SERVER)
  c4_type = computation_types.FederatedType([
      newly_determined_types['c5_type'].member,
      previously_packed_types['c6_type'].member
  ], placements.CLIENTS)
  newly_determined_types['c4_type'] = c4_type
  newly_determined_types['work_type'] = computation_types.FunctionType(
      c3_type.member, c4_type.member)
  return dict(
      itertools.chain(previously_packed_types.items(),
                      newly_determined_types.items()))


def check_and_pack_after_aggregate_type_signature(type_spec,
                                                  previously_packed_types):
  """Checks types inferred from `after_aggregate` and packs in `previously_packed_types`.

  After splitting the `next` portion of a `tff.utils.IterativeProcess` all the
  way down, `after_aggregate` should have
  type signature `<<<s1,c1>,c2>,s3> -> <s6,s7,c6>`. This
  function validates every element of the above, extracting and packing in
  addition types of `s3` and `s4`.

  Args:
    type_spec: The `type_signature` attribute of the `after_aggregate` portion
      of the `tff.utils.IterativeProcess` from which we are looking to extract
      an instance of `canonical_form.CanonicalForm`.
    previously_packed_types: Dict containing the information from `next`,
      `before_broadcast` and `before_aggregate` in the iterative process we are
      parsing.

  Returns:
    A `dict` packing the types which can be inferred from `type_spec`.

  Raises:
    TypeError: If `type_signature` is incompatible with
    `previously_packed_types`.
  """
  should_raise = False
  if not (type_spec.parameter[0][0][0] == previously_packed_types['s1_type'] and
          type_spec.parameter[0][0][1] == previously_packed_types['c1_type'] and
          type_spec.parameter[0][1] == previously_packed_types['c2_type'] and
          type_spec.parameter[1] == previously_packed_types['s3_type']):
    should_raise = True
  if not (type_spec.result[0] == previously_packed_types['s6_type'] and
          type_spec.result[1] == previously_packed_types['s7_type']):
    should_raise = True
  if len(type_spec.result
        ) == 3 and type_spec.result[2] != previously_packed_types['c6_type']:
    should_raise = True
  if should_raise:
    # TODO(b/121290421): These error messages, and indeed the 'track boolean and
    # raise once' logic of these methods as well, is intended to be provisional
    # and revisited when we've seen the compilation pipeline fail more clearly,
    # or maybe preferably iteratively improved as new failure modes are
    # encountered.
    raise TypeError(
        'Encountered a type error while checking `after_aggregate`; '
        'expected a type signature of the form '
        '`<<<s1,c1>,c2>,s3> -> <s6,s7,c6>`, where s1 matches {}, '
        'c1 matches {}, c2 matches {}, s3 matches {}, s6 matches '
        '{}, s7 matches {}, c6 matches {}, as defined in '
        '`canonical_form.CanonicalForm`. Encountered a type signature '
        '{}.'.format(previously_packed_types['s1_type'],
                     previously_packed_types['c1_type'],
                     previously_packed_types['c2_type'],
                     previously_packed_types['s3_type'],
                     previously_packed_types['s6_type'],
                     previously_packed_types['s7_type'],
                     previously_packed_types['c6_type'], type_spec))
  s4_type = computation_types.FederatedType([
      previously_packed_types['s1_type'].member,
      previously_packed_types['s3_type'].member
  ], placements.SERVER)
  s5_type = computation_types.FederatedType([
      previously_packed_types['s6_type'].member,
      previously_packed_types['s7_type'].member
  ], placements.SERVER)
  newly_determined_types = {}
  newly_determined_types['s4_type'] = s4_type
  newly_determined_types['s5_type'] = s5_type
  newly_determined_types['update_type'] = computation_types.FunctionType(
      s4_type.member, s5_type.member)
  c3_type = computation_types.FederatedType([
      previously_packed_types['c1_type'].member,
      previously_packed_types['c2_type'].member
  ], placements.CLIENTS)
  newly_determined_types['c3_type'] = c3_type
  return dict(
      itertools.chain(previously_packed_types.items(),
                      newly_determined_types.items()))


def extract_prepare(before_broadcast):
  """Converts `before_broadcast` into `prepare`.

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
  # See `get_iterative_process_for_canonical_form()` above for the meaning of
  # variable names used in the code below.
  s1_index_in_before_broadcast = 0
  s1_to_s2_computation = (
      transformations.bind_single_selection_as_argument_to_lower_level_lambda(
          before_broadcast, s1_index_in_before_broadcast)).result.function
  prepare = transformations.consolidate_and_extract_local_processing(
      s1_to_s2_computation)
  return prepare


def extract_work(before_aggregate, after_aggregate):
  """Converts `before_aggregate` and `after_aggregate` to `work`.

  Args:
    before_aggregate: The first result of splitting `after_broadcast` on
      `intrinsic_defs.FEDERATED_AGGREGATE`.
    after_aggregate: The second result of splitting `after_broadcast` on
      `intrinsic_defs.FEDERATED_AGGREGATE`.

  Returns:
    `work` as specified by `canonical_form.CanonicalForm`, an instance of
    `building_blocks.CompiledComputation`.

  Raises:
    transformations.CanonicalFormCompilationError: If we extract an AST of the
      wrong type.
  """
  # See `get_iterative_process_for_canonical_form()` above for the meaning of
  # variable names used in the code below.
  c3_elements_in_before_aggregate_parameter = [[0, 1], [1]]
  c3_to_before_aggregate_computation = (
      transformations.zip_selection_as_argument_to_lower_level_lambda(
          before_aggregate,
          c3_elements_in_before_aggregate_parameter).result.function)
  c5_index_in_before_aggregate_result = 0
  c3_to_c5_computation = transformations.select_output_from_lambda(
      c3_to_before_aggregate_computation, c5_index_in_before_aggregate_result)
  c6_index_in_after_aggregate_result = 2
  after_aggregate_to_c6_computation = transformations.select_output_from_lambda(
      after_aggregate, c6_index_in_after_aggregate_result)
  c3_elements_in_after_aggregate_parameter = [[0, 0, 1], [0, 1]]
  c3_to_c6_computation = (
      transformations.zip_selection_as_argument_to_lower_level_lambda(
          after_aggregate_to_c6_computation,
          c3_elements_in_after_aggregate_parameter).result.function)
  c3_to_unzipped_c4_computation = transformations.concatenate_function_outputs(
      c3_to_c5_computation, c3_to_c6_computation)
  c3_to_c4_computation = building_blocks.Lambda(
      c3_to_unzipped_c4_computation.parameter_name,
      c3_to_unzipped_c4_computation.parameter_type,
      building_block_factory.create_federated_zip(
          c3_to_unzipped_c4_computation.result))

  work = transformations.consolidate_and_extract_local_processing(
      c3_to_c4_computation)
  return work


def extract_aggregate_functions(before_aggregate):
  """Converts `before_aggregate` to aggregation functions.

  Args:
    before_aggregate: The first result of splitting `after_broadcast` on
      `intrinsic_defs.FEDERATED_AGGREGATE`.

  Returns:
    `zero`, `accumulate`, `merge` and `report` as specified by
    `canonical_form.CanonicalForm`. All are instances of
    `building_blocks.CompiledComputation`.

  Raises:
    transformations.CanonicalFormCompilationError: If we extract an ASTs of the
      wrong type.
  """
  # See `get_iterative_process_for_canonical_form()` above for the meaning of
  # variable names used in the code below.
  zero_index_in_before_aggregate_result = 1
  zero_tff = transformations.select_output_from_lambda(
      before_aggregate, zero_index_in_before_aggregate_result).result
  accumulate_index_in_before_aggregate_result = 2
  accumulate_tff = transformations.select_output_from_lambda(
      before_aggregate, accumulate_index_in_before_aggregate_result).result
  merge_index_in_before_aggregate_result = 3
  merge_tff = transformations.select_output_from_lambda(
      before_aggregate, merge_index_in_before_aggregate_result).result
  report_index_in_before_aggregate_result = 4
  report_tff = transformations.select_output_from_lambda(
      before_aggregate, report_index_in_before_aggregate_result).result

  zero = transformations.consolidate_and_extract_local_processing(zero_tff)
  accumulate = transformations.consolidate_and_extract_local_processing(
      accumulate_tff)
  merge = transformations.consolidate_and_extract_local_processing(merge_tff)
  report = transformations.consolidate_and_extract_local_processing(report_tff)
  return zero, accumulate, merge, report


def extract_update(after_aggregate):
  """Converts `after_aggregate` to `update`.

  Args:
    after_aggregate: The second result of splitting `after_broadcast` on
      `intrinsic_defs.FEDERATED_AGGREGATE`.

  Returns:
    `update` as specified by `canonical_form.CanonicalForm`, an instance of
    `building_blocks.CompiledComputation`.

  Raises:
    transformations.CanonicalFormCompilationError: If we extract an AST of the
      wrong type.
  """
  # See `get_iterative_process_for_canonical_form()` above for the meaning of
  # variable names used in the code below.
  s5_elements_in_after_aggregate_result = [0, 1]
  s5_output_extracted = transformations.select_output_from_lambda(
      after_aggregate, s5_elements_in_after_aggregate_result)
  s5_output_zipped = building_blocks.Lambda(
      s5_output_extracted.parameter_name, s5_output_extracted.parameter_type,
      building_block_factory.create_federated_zip(s5_output_extracted.result))
  s4_elements_in_after_aggregate_parameter = [[0, 0, 0], [1]]
  s4_to_s5_computation = (
      transformations.zip_selection_as_argument_to_lower_level_lambda(
          s5_output_zipped,
          s4_elements_in_after_aggregate_parameter).result.function)

  update = transformations.consolidate_and_extract_local_processing(
      s4_to_s5_computation)
  return update


def replace_intrinsics_with_bodies(comp):
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


def get_canonical_form_for_iterative_process(iterative_process):
  """Constructs `tff.backends.mapreduce.CanonicalForm` given iterative process.

  This function transforms computations from the input `iterative_process` into
  an instance of `tff.backends.mapreduce.CanonicalForm`.

  Args:
    iterative_process: An instance of `tff.utils.IterativeProcess`.

  Returns:
    An instance of `tff.backends.mapreduce.CanonicalForm` equivalent to this
    process.

  Raises:
    TypeError: If the arguments are of the wrong types.
    transformations.CanonicalFormCompilationError: If the compilation
      process fails.
  """
  py_typecheck.check_type(iterative_process, computation_utils.IterativeProcess)

  initialize_comp = building_blocks.ComputationBuildingBlock.from_proto(
      iterative_process.initialize._computation_proto)  # pylint: disable=protected-access

  next_comp = building_blocks.ComputationBuildingBlock.from_proto(
      iterative_process.next._computation_proto)  # pylint: disable=protected-access

  if not (isinstance(next_comp.type_signature.parameter,
                     computation_types.NamedTupleType) and
          isinstance(next_comp.type_signature.result,
                     computation_types.NamedTupleType)):
    raise TypeError(
        'Any IterativeProcess compatible with CanonicalForm must '
        'have a `next` function which takes and returns instances '
        'of `tff.NamedTupleType`; your next function takes '
        'parameters of type {} and returns results of type {}'.format(
            next_comp.type_signature.parameter,
            next_comp.type_signature.result))

  if len(next_comp.type_signature.result) == 2:
    next_comp = _create_next_with_fake_client_output(next_comp)

  initialize_comp = replace_intrinsics_with_bodies(initialize_comp)
  next_comp = replace_intrinsics_with_bodies(next_comp)

  tree_analysis.check_intrinsics_whitelisted_for_reduction(initialize_comp)
  tree_analysis.check_intrinsics_whitelisted_for_reduction(next_comp)
  tree_analysis.check_broadcast_not_dependent_on_aggregate(next_comp)

  if tree_analysis.contains_called_intrinsic(
      next_comp, intrinsic_defs.FEDERATED_BROADCAST.uri):
    before_broadcast, after_broadcast = (
        transformations.force_align_and_split_by_intrinsics(
            next_comp, [intrinsic_defs.FEDERATED_BROADCAST.uri]))
  else:
    before_broadcast, after_broadcast = (
        _create_before_and_after_broadcast_for_no_broadcast(next_comp))

  before_aggregate, after_aggregate = (
      transformations.force_align_and_split_by_intrinsics(
          after_broadcast, [intrinsic_defs.FEDERATED_AGGREGATE.uri]))

  type_info = pack_initialize_comp_type_signature(
      initialize_comp.type_signature)

  type_info = pack_next_comp_type_signature(next_comp.type_signature, type_info)

  type_info = check_and_pack_before_broadcast_type_signature(
      before_broadcast.type_signature, type_info)

  type_info = check_and_pack_before_aggregate_type_signature(
      before_aggregate.type_signature, type_info)

  type_info = check_and_pack_after_aggregate_type_signature(
      after_aggregate.type_signature, type_info)

  initialize = transformations.consolidate_and_extract_local_processing(
      initialize_comp)
  _check_type_equal(initialize.type_signature, type_info['initialize_type'],
                    'initialize')

  prepare = extract_prepare(before_broadcast)
  _check_type_equal(prepare.type_signature, type_info['prepare_type'],
                    'prepare')

  work = extract_work(before_aggregate, after_aggregate)
  _check_type_equal(work.type_signature, type_info['work_type'], 'work')

  zero, accumulate, merge, report = extract_aggregate_functions(
      before_aggregate)
  _check_type_equal(zero.type_signature, type_info['zero_type'], 'zero')
  _check_type_equal(accumulate.type_signature, type_info['accumulate_type'],
                    'accumulate')
  _check_type_equal(merge.type_signature, type_info['merge_type'], 'merge')
  _check_type_equal(report.type_signature, type_info['report_type'], 'report')

  update = extract_update(after_aggregate)
  _check_type_equal(update.type_signature, type_info['update_type'], 'update')

  return canonical_form.CanonicalForm(
      computation_wrapper_instances.building_block_to_computation(initialize),
      computation_wrapper_instances.building_block_to_computation(prepare),
      computation_wrapper_instances.building_block_to_computation(work),
      computation_wrapper_instances.building_block_to_computation(zero),
      computation_wrapper_instances.building_block_to_computation(accumulate),
      computation_wrapper_instances.building_block_to_computation(merge),
      computation_wrapper_instances.building_block_to_computation(report),
      computation_wrapper_instances.building_block_to_computation(update))


def _create_next_with_fake_client_output(tree):
  """Creates a next computation with a fake client output.

  This function is intended to be used by
  `get_canonical_form_for_iterative_process` to create a next computation with
  a fake client output when no client output is returned by the `next` function
  of the `tff.utils.IterativeProcess`.

  NOTE: This function does not assert that there is no client output in `tree`,
  the caller is expected to perform this check before calling this function.

  Args:
    tree: An instance of `building_blocks.ComputationBuildingBlock`.

  Returns:
    A new `building_blocks.ComputationBuildingBlock` representing a next
    computaiton with a fake client output.
  """
  if isinstance(tree.result, building_blocks.Tuple):
    arg_1 = tree.result[0]
    arg_2 = tree.result[1]
  else:
    arg_1 = building_blocks.Selection(tree.result, index=0)
    arg_2 = building_blocks.Selection(tree.result, index=1)

  empty_tuple = building_blocks.Tuple([])
  client_output = building_block_factory.create_federated_value(
      empty_tuple, placements.CLIENTS)
  output = building_blocks.Tuple([arg_1, arg_2, client_output])
  return building_blocks.Lambda(tree.parameter_name, tree.parameter_type,
                                output)


def _create_before_and_after_broadcast_for_no_broadcast(tree):
  """Creates a before and after broadcast computations for the given `tree`.

  This function is intended to be used by
  `get_canonical_form_for_iterative_process` to create before and after
  broadcast computations for the given `tree` when there is no
  `intrinsic_defs.FEDERATED_BROADCAST` in `tree`.

  NOTE: This function does not assert that there is no
  `intrinsic_defs.FEDERATED_BROADCAST` in `tree`, the caller is expected to
  perform this check before calling this function.

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
  empty_tuple = building_blocks.Tuple([])
  value = building_block_factory.create_federated_value(empty_tuple,
                                                        placements.SERVER)
  before_broadcast = building_blocks.Lambda(parameter_name,
                                            tree.type_signature.parameter,
                                            value)

  parameter_name = next(name_generator)
  type_signature = computation_types.FederatedType(
      before_broadcast.type_signature.result.member, placements.CLIENTS)
  parameter_type = computation_types.NamedTupleType(
      [tree.type_signature.parameter, type_signature])
  ref = building_blocks.Reference(parameter_name, parameter_type)
  arg = building_blocks.Selection(ref, index=0)
  call = building_blocks.Call(tree, arg)
  after_broadcast = building_blocks.Lambda(ref.name, ref.type_signature, call)

  return before_broadcast, after_broadcast

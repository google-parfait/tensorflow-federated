# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import six

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core import api as tff
from tensorflow_federated.python.core import framework as tff_framework
from tensorflow_federated.python.core import utils as tff_utils
from tensorflow_federated.python.core.backends.mapreduce import canonical_form
from tensorflow_federated.python.core.backends.mapreduce import transformations


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

  @tff.federated_computation
  def init_computation():
    return tff.federated_value(cf.initialize(), tff.SERVER)

  @tff.federated_computation(
      init_computation.type_signature.result,
      tff.FederatedType(cf.work.type_signature.parameter[0], tff.CLIENTS))
  def next_computation(arg):
    """The logic of a single MapReduce sprocessing round."""
    s1 = arg[0]
    c1 = arg[1]
    s2 = tff.federated_apply(cf.prepare, s1)
    c2 = tff.federated_broadcast(s2)
    c3 = tff.federated_zip([c1, c2])
    c4 = tff.federated_map(cf.work, c3)
    c5 = c4[0]
    c6 = c4[1]
    s3 = tff.federated_aggregate(
        c5, cf.zero(), cf.accumulate, cf.merge, cf.report)
    s4 = tff.federated_zip([s1, s3])
    s5 = tff.federated_apply(cf.update, s4)
    s6 = s5[0]
    s7 = s5[1]
    return s6, s7, c6

  return tff_utils.IterativeProcess(init_computation, next_computation)


def pack_initialize_comp_type_signature(type_spec):
  """Packs the initialize type to be used by the remainder of the compiler."""
  if not (isinstance(type_spec, tff.FederatedType) and
          type_spec.placement == tff.SERVER):
    raise TypeError(
        'Expected init type spec to be a federated type placed at the server; '
        'instead found {}'.format(type_spec))
  return {'initialize_type': type_spec}


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
  if not (isinstance(type_signature, tff.FunctionType) and
          isinstance(type_signature.parameter, tff.NamedTupleType) and
          len(type_signature.parameter) == 2 and
          isinstance(type_signature.result, tff.NamedTupleType) and
          len(type_signature.result) == 3):
    should_raise = True
  if type_signature.parameter[0] != previously_packed_types['initialize_type']:
    should_raise = True
  for server_placed_type in [
      type_signature.parameter[0], type_signature.result[0],
      type_signature.result[1]
  ]:
    if not (isinstance(server_placed_type, tff.FederatedType) and
            server_placed_type.placement == tff.SERVER):
      should_raise = True

  if len(type_signature.result) == 3:
    for client_placed_type in [
        type_signature.parameter[1], type_signature.result[2]
    ]:
      if not (isinstance(client_placed_type, tff.FederatedType) and
              client_placed_type.placement == tff.CLIENTS):
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
      itertools.chain(
          six.iteritems(previously_packed_types),
          six.iteritems(newly_determined_types)))


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
  if not (isinstance(type_spec, tff.FunctionType) and
          isinstance(type_spec.parameter, tff.NamedTupleType) and
          len(type_spec.parameter) == 2 and
          type_spec.parameter[0] == previously_packed_types['s1_type'] and
          type_spec.parameter[1] == previously_packed_types['c1_type']):
    should_raise = True
  if not (isinstance(type_spec.result, tff.FederatedType) and
          type_spec.result.placement == tff.SERVER):
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
  newly_determined_types['prepare_type'] = tff.FunctionType(
      previously_packed_types['s1_type'].member, s2.member)
  return dict(
      itertools.chain(
          six.iteritems(previously_packed_types),
          six.iteritems(newly_determined_types)))


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
  if not (isinstance(type_spec, tff.FunctionType) and
          isinstance(type_spec.parameter, tff.NamedTupleType)):
    should_raise = True
  if not (isinstance(type_spec.parameter[0], tff.NamedTupleType) and
          len(type_spec.parameter[0]) == 2 and
          type_spec.parameter[0][0] == previously_packed_types['s1_type'] and
          type_spec.parameter[0][1] == previously_packed_types['c1_type']):
    should_raise = True
  if not (isinstance(type_spec.parameter[1], tff.FederatedType) and
          type_spec.parameter[1].placement == tff.CLIENTS and type_spec
          .parameter[1].member == previously_packed_types['s2_type'].member):
    should_raise = True
  if not (
      isinstance(type_spec.result, tff.NamedTupleType) and
      len(type_spec.result) == 5 and
      isinstance(type_spec.result[0], tff.FederatedType) and
      type_spec.result[0].placement == tff.CLIENTS and
      tff_framework.is_tensorflow_compatible_type(type_spec.result[1]) and
      type_spec.result[2] == tff.FunctionType([
          type_spec.result[1], type_spec.result[0].member
      ], type_spec.result[1]) and type_spec.result[3] == tff.FunctionType(
          [type_spec.result[1], type_spec.result[1]], type_spec.result[1]) and
      type_spec.result[4].parameter == type_spec.result[1] and
      tff_framework.is_tensorflow_compatible_type(type_spec.result[4].result)):
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
  c3_type = tff.FederatedType(
      [previously_packed_types['c1_type'].member, c2_type.member], tff.CLIENTS)
  newly_determined_types['c3_type'] = c3_type
  c5_type = type_spec.result[0]
  zero_type = tff.FunctionType(None, type_spec.result[1])
  accumulate_type = type_spec.result[2]
  merge_type = type_spec.result[3]
  report_type = type_spec.result[4]
  newly_determined_types['c5_type'] = c5_type
  newly_determined_types['zero_type'] = zero_type
  newly_determined_types['accumulate_type'] = accumulate_type
  newly_determined_types['merge_type'] = merge_type
  newly_determined_types['report_type'] = report_type
  newly_determined_types['s3_type'] = tff.FederatedType(report_type.result,
                                                        tff.SERVER)
  c4_type = tff.FederatedType([
      newly_determined_types['c5_type'].member,
      previously_packed_types['c6_type'].member
  ], tff.CLIENTS)
  newly_determined_types['c4_type'] = c4_type
  newly_determined_types['work_type'] = tff.FunctionType(
      c3_type.member, c4_type.member)
  return dict(
      itertools.chain(
          six.iteritems(previously_packed_types),
          six.iteritems(newly_determined_types)))


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
  s4_type = tff.FederatedType([
      previously_packed_types['s1_type'].member,
      previously_packed_types['s3_type'].member
  ], tff.SERVER)
  s5_type = tff.FederatedType([
      previously_packed_types['s6_type'].member,
      previously_packed_types['s7_type'].member
  ], tff.SERVER)
  newly_determined_types = {}
  newly_determined_types['s4_type'] = s4_type
  newly_determined_types['s5_type'] = s5_type
  newly_determined_types['update_type'] = tff.FunctionType(
      s4_type.member, s5_type.member)
  c3_type = tff.FederatedType([
      previously_packed_types['c1_type'].member,
      previously_packed_types['c2_type'].member
  ], tff.CLIENTS)
  newly_determined_types['c3_type'] = c3_type
  return dict(
      itertools.chain(
          six.iteritems(previously_packed_types),
          six.iteritems(newly_determined_types)))


def extract_prepare(before_broadcast, canonical_form_types):
  """Converts `before_broadcast` into `prepare`.

  Args:
    before_broadcast: The first result of splitting `next_comp` on
      `tff_framework.FEDERATED_BROADCAST`.
    canonical_form_types: `dict` holding the `canonical_form.CanonicalForm` type
      signatures specified by the `tff.utils.IterativeProcess` we are compiling.

  Returns:
    `prepare` as specified by `canonical_form.CanonicalForm`, an instance of
    `tff_framework.CompiledComputation`.

  Raises:
    transformations.CanonicalFormCompilationError: If we fail to extract a
    `tff_framework.CompiledComputation`, or we extract one of the wrong type.
  """
  # See `get_iterative_process_for_canonical_form()` above for the meaning of
  # variable names used in the code below.
  s1_index_in_before_broadcast = 0
  s1_to_s2_computation = (
      transformations.bind_single_selection_as_argument_to_lower_level_lambda(
          before_broadcast, s1_index_in_before_broadcast)).result.function
  prepare = transformations.consolidate_and_extract_local_processing(
      s1_to_s2_computation)
  if not isinstance(prepare, tff_framework.CompiledComputation):
    raise transformations.CanonicalFormCompilationError(
        'Failed to extract a `tff_framework.CompiledComputation` from '
        'prepare, instead received a {} (of type {}).'.format(
            type(prepare), prepare.type_signature))
  if prepare.type_signature != canonical_form_types['prepare_type']:
    raise transformations.CanonicalFormCompilationError(
        'Extracted a TF block of the wrong type. Expected a function with type '
        '{}, but the type signature of the TF block was {}'.format(
            canonical_form_types['prepare_type'], prepare.type_signature))
  return prepare


def extract_work(before_aggregate, after_aggregate, canonical_form_types):
  """Converts `before_aggregate` and `after_aggregate` to `work`.

  Args:
    before_aggregate: The first result of splitting `after_broadcast` on
      `tff_framework.FEDERATED_AGGREGATE`.
    after_aggregate: The second result of splitting `after_broadcast` on
      `tff_framework.FEDERATED_AGGREGATE`.
    canonical_form_types: `dict` holding the `canonical_form.CanonicalForm` type
      signatures specified by the `tff.utils.IterativeProcess` we are compiling.

  Returns:
    `work` as specified by `canonical_form.CanonicalForm`, an instance of
    `tff_framework.CompiledComputation`.

  Raises:
    transformations.CanonicalFormCompilationError: If we fail to extract a
    `tff_framework.CompiledComputation`, or we extract one of the wrong type.
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
  c3_to_c4_computation = tff_framework.Lambda(
      c3_to_unzipped_c4_computation.parameter_name,
      c3_to_unzipped_c4_computation.parameter_type,
      tff_framework.create_federated_zip(c3_to_unzipped_c4_computation.result))

  work = transformations.consolidate_and_extract_local_processing(
      c3_to_c4_computation)
  if not isinstance(work, tff_framework.CompiledComputation):
    raise transformations.CanonicalFormCompilationError(
        'Failed to extract a `tff_framework.CompiledComputation` from '
        'work, instead received a {} (of type {}).'.format(
            type(work), work.type_signature))
  if work.type_signature != canonical_form_types['work_type']:
    raise transformations.CanonicalFormCompilationError(
        'Extracted a TF block of the wrong type. Expected a function with type '
        '{}, but the type signature of the TF block was {}'.format(
            canonical_form_types['work_type'], work.type_signature))
  return work


def extract_aggregate_functions(before_aggregate, canonical_form_types):
  """Converts `before_aggregate` to aggregation functions.

  Args:
    before_aggregate: The first result of splitting `after_broadcast` on
      `tff_framework.FEDERATED_AGGREGATE`.
    canonical_form_types: `dict` holding the `canonical_form.CanonicalForm` type
      signatures specified by the `tff.utils.IterativeProcess` we are compiling.

  Returns:
    `zero`, `accumulate`, `merge` and `report` as specified by
    `canonical_form.CanonicalForm`. All are instances of
    `tff_framework.CompiledComputation`.

  Raises:
    transformations.CanonicalFormCompilationError: if we fail to extract
    `tff_framework.CompiledComputation`s, or we extract one of the wrong type.
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
  for name, tf_block in (('zero', zero), ('accumulate', accumulate),
                         ('merge', merge), ('report', report)):
    if not isinstance(tf_block, tff_framework.CompiledComputation):
      raise transformations.CanonicalFormCompilationError(
          'Failed to extract a `tff_framework.CompiledComputation` from '
          '{}, instead received a {} (of type {}).'.format(
              name, type(tf_block), tf_block.type_signature))
    if tf_block.type_signature != canonical_form_types['{}_type'.format(name)]:
      raise transformations.CanonicalFormCompilationError(
          'Extracted a TF block of the wrong type. Expected a function with type '
          '{}, but the type signature of the TF block was {}'.format(
              canonical_form_types['{}_type'.format(name)],
              tf_block.type_signature))
  return zero, accumulate, merge, report


def extract_update(after_aggregate, canonical_form_types):
  """Converts `after_aggregate` to `update`.

  Args:
    after_aggregate: The second result of splitting `after_broadcast` on
      `tff_framework.FEDERATED_AGGREGATE`.
    canonical_form_types: `dict` holding the `canonical_form.CanonicalForm` type
      signatures specified by the `tff.utils.IterativeProcess` we are compiling.

  Returns:
    `update` as specified by `canonical_form.CanonicalForm`, an instance of
    `tff_framework.CompiledComputation`.

  Raises:
    transformations.CanonicalFormCompilationError: If we fail to extract a
    `tff_framework.CompiledComputation`, or we extract one of the wrong type.
  """
  # See `get_iterative_process_for_canonical_form()` above for the meaning of
  # variable names used in the code below.
  s5_elements_in_after_aggregate_result = [0, 1]
  s5_output_extracted = transformations.select_output_from_lambda(
      after_aggregate, s5_elements_in_after_aggregate_result)
  s5_output_zipped = tff_framework.Lambda(
      s5_output_extracted.parameter_name, s5_output_extracted.parameter_type,
      tff_framework.create_federated_zip(s5_output_extracted.result))
  s4_elements_in_after_aggregate_parameter = [[0, 0, 0], [1]]
  s4_to_s5_computation = (
      transformations.zip_selection_as_argument_to_lower_level_lambda(
          s5_output_zipped,
          s4_elements_in_after_aggregate_parameter).result.function)

  update = transformations.consolidate_and_extract_local_processing(
      s4_to_s5_computation)
  if not isinstance(update, tff_framework.CompiledComputation):
    raise transformations.CanonicalFormCompilationError(
        'Failed to extract a `tff_framework.CompiledComputation` from '
        'update, instead received a {} (of type {}).'.format(
            type(update), update.type_signature))
  if update.type_signature != canonical_form_types['update_type']:
    raise transformations.CanonicalFormCompilationError(
        'Extracted a TF block of the wrong type. Expected a function with type '
        '{}, but the type signature of the TF block was {}'.format(
            canonical_form_types['update_type'], update.type_signature))
  return update


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
  py_typecheck.check_type(iterative_process, tff_utils.IterativeProcess)

  initialize_comp = tff_framework.ComputationBuildingBlock.from_proto(
      iterative_process.initialize._computation_proto)  # pylint: disable=protected-access

  next_comp = tff_framework.ComputationBuildingBlock.from_proto(
      iterative_process.next._computation_proto)  # pylint: disable=protected-access

  if len(next_comp.type_signature.result) == 2:
    next_result = next_comp.result
    dummy_clients_metrics_appended = tff_framework.Tuple([
        next_result[0],
        next_result[1],
        tff.federated_value([], tff.CLIENTS)._comp  # pylint: disable=protected-access
    ])
    next_comp = tff_framework.Lambda(next_comp.parameter_name,
                                     next_comp.parameter_type,
                                     dummy_clients_metrics_appended)

  initialize_comp = tff_framework.replace_intrinsics_with_bodies(
      initialize_comp)
  next_comp = tff_framework.replace_intrinsics_with_bodies(next_comp)

  tff_framework.check_intrinsics_whitelisted_for_reduction(initialize_comp)
  tff_framework.check_intrinsics_whitelisted_for_reduction(next_comp)
  tff_framework.check_broadcast_not_dependent_on_aggregate(next_comp)

  before_broadcast, after_broadcast = (
      transformations.force_align_and_split_by_intrinsic(
          next_comp, tff_framework.FEDERATED_BROADCAST.uri))

  before_aggregate, after_aggregate = (
      transformations.force_align_and_split_by_intrinsic(
          after_broadcast, tff_framework.FEDERATED_AGGREGATE.uri))

  init_info_packed = pack_initialize_comp_type_signature(
      initialize_comp.type_signature)

  next_info_packed = pack_next_comp_type_signature(next_comp.type_signature,
                                                   init_info_packed)

  before_broadcast_info_packed = (
      check_and_pack_before_broadcast_type_signature(
          before_broadcast.type_signature, next_info_packed))

  before_aggregate_info_packed = (
      check_and_pack_before_aggregate_type_signature(
          before_aggregate.type_signature, before_broadcast_info_packed))

  canonical_form_types = check_and_pack_after_aggregate_type_signature(
      after_aggregate.type_signature, before_aggregate_info_packed)

  initialize = transformations.consolidate_and_extract_local_processing(
      initialize_comp)

  if not (isinstance(initialize, tff_framework.CompiledComputation) and
          initialize.type_signature.result ==
          canonical_form_types['initialize_type'].member):
    raise transformations.CanonicalFormCompilationError(
        'Compilation of initialize has failed. Expected to extract a '
        '`tff_framework.CompiledComputation` of type {}, instead we extracted '
        'a {} of type {}.'.format(next_comp.type_signature.parameter[0],
                                  type(initialize),
                                  initialize.type_signature.result))

  prepare = extract_prepare(before_broadcast, canonical_form_types)

  work = extract_work(before_aggregate, after_aggregate, canonical_form_types)

  zero_noarg_function, accumulate, merge, report = extract_aggregate_functions(
      before_aggregate, canonical_form_types)

  update = extract_update(after_aggregate, canonical_form_types)

  cf = canonical_form.CanonicalForm(
      tff_framework.building_block_to_computation(initialize),
      tff_framework.building_block_to_computation(prepare),
      tff_framework.building_block_to_computation(work),
      tff_framework.building_block_to_computation(zero_noarg_function),
      tff_framework.building_block_to_computation(accumulate),
      tff_framework.building_block_to_computation(merge),
      tff_framework.building_block_to_computation(report),
      tff_framework.building_block_to_computation(update))
  return cf

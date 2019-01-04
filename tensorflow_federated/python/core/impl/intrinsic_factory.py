# Copyright 2018, The TensorFlow Federated Authors.
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
"""A factory of intrinsics for use in composing federated computations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import value_base
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import type_constructors
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl import value_impl
from tensorflow_federated.python.core.impl import value_utils


class IntrinsicFactory(object):
  """A factory that can constructs intrinsics over the given context stack."""

  # NOTE: Tests for this class currently go to `api/intrinsics_test.py`.

  def __init__(self, context_stack):
    """Constructs this factory over the given context stack.

    Args:
      context_stack: The context stack to use.
    """
    py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
    self._context_stack = context_stack

  def federated_aggregate(self, value, zero, accumulate, merge, report):
    """Implements `federated_aggregate` as defined in `api/intrinsics.py`.

    Args:
      value: As in `api/intrinsics.py`.
      zero: As in `api/intrinsics.py`.
      accumulate: As in `api/intrinsics.py`.
      merge: As in `api/intrinsics.py`.
      report: As in `api/intrinsics.py`.

    Returns:
      As in `api/intrinsics.py`.

    Raises:
      TypeError: As in `api/intrinsics.py`.
    """
    value = value_impl.to_value(value, None, self._context_stack)
    type_utils.check_federated_value_placement(value, placements.CLIENTS,
                                               'value to be aggregated')

    zero = value_impl.to_value(zero, None, self._context_stack)
    py_typecheck.check_type(zero, value_base.Value)

    # TODO(b/113112108): We need a check here that zero does not have federated
    # constituents.

    accumulate = value_impl.to_value(accumulate, None, self._context_stack)
    merge = value_impl.to_value(merge, None, self._context_stack)
    report = value_impl.to_value(report, None, self._context_stack)
    for op in [accumulate, merge, report]:
      py_typecheck.check_type(op, value_base.Value)
      py_typecheck.check_type(op.type_signature, computation_types.FunctionType)

    accumulate_type_expected = type_constructors.reduction_op(
        zero.type_signature, value.type_signature.member)
    merge_type_expected = type_constructors.reduction_op(
        zero.type_signature, zero.type_signature)
    report_type_expected = computation_types.FunctionType(
        zero.type_signature, report.type_signature.result)
    for op_name, op, type_expected in [('accumulate', accumulate,
                                        accumulate_type_expected),
                                       ('merge', merge, merge_type_expected),
                                       ('report', report,
                                        report_type_expected)]:
      if not type_utils.is_assignable_from(type_expected, op.type_signature):
        raise TypeError('Expected parameter `{}` to be of type {}, '
                        'but received {} instead.'.format(
                            op_name, str(type_expected),
                            str(op.type_signature)))

    result_type = computation_types.FederatedType(report.type_signature.result,
                                                  placements.SERVER, True)
    intrinsic = value_impl.ValueImpl(
        computation_building_blocks.Intrinsic(
            intrinsic_defs.FEDERATED_AGGREGATE.uri,
            computation_types.FunctionType([
                value.type_signature, zero.type_signature,
                accumulate_type_expected, merge_type_expected,
                report_type_expected
            ], result_type)), self._context_stack)
    return intrinsic(value, zero, accumulate, merge, report)

  def federated_average(self, value, weight):
    """Implements `federated_average` as defined in `api/intrinsics.py`.

    Args:
      value: As in `api/intrinsics.py`.
      weight: As in `api/intrinsics.py`.

    Returns:
      As in `api/intrinsics.py`.

    Raises:
      TypeError: As in `api/intrinsics.py`.
    """
    # TODO(b/113112108): Possibly relax the constraints on numeric types, and
    # inject implicit casts where appropriate. For instance, we might want to
    # allow `tf.int32` values as the input, and automatically cast them to
    # `tf.float321 before invoking the average, thus producing a floating-point
    # result.

    # TODO(b/120439632): Possibly allow the weight to be either structured or
    # non-scalar, e.g., for the case of averaging a convolutional layer, when
    # we would want to use a different weight for every filter, and where it
    # might be cumbersome for users to have to manually slice and assemble a
    # variable.

    value = value_impl.to_value(value, None, self._context_stack)
    type_utils.check_federated_value_placement(value, placements.CLIENTS,
                                               'value to be averaged')
    if not type_utils.is_average_compatible(value.type_signature):
      raise TypeError(
          'The value type {} is not compatible with the average operator.'
          .format(str(value.type_signature)))

    if weight is not None:
      weight = value_impl.to_value(weight, None, self._context_stack)
      type_utils.check_federated_value_placement(weight, placements.CLIENTS,
                                                 'weight to use in averaging')
      py_typecheck.check_type(weight.type_signature.member,
                              computation_types.TensorType)
      if weight.type_signature.member.shape.ndims != 0:
        raise TypeError('The weight type {} is not a federated scalar.'.format(
            str(weight.type_signature)))
      if not (weight.type_signature.member.dtype.is_integer or
              weight.type_signature.member.dtype.is_floating):
        raise TypeError('The weight type {} is not a federated integer or '
                        'floating-point tensor.'.format(
                            str(weight.type_signature)))

    result_type = computation_types.FederatedType(value.type_signature.member,
                                                  placements.SERVER, True)

    if weight is not None:
      intrinsic = value_impl.ValueImpl(
          computation_building_blocks.Intrinsic(
              intrinsic_defs.FEDERATED_WEIGHTED_AVERAGE.uri,
              computation_types.FunctionType(
                  [value.type_signature, weight.type_signature], result_type)),
          self._context_stack)
      return intrinsic(value, weight)
    else:
      intrinsic = value_impl.ValueImpl(
          computation_building_blocks.Intrinsic(
              intrinsic_defs.FEDERATED_AVERAGE.uri,
              computation_types.FunctionType(value.type_signature,
                                             result_type)), self._context_stack)
      return intrinsic(value)

  def federated_broadcast(self, value):
    """Implements `federated_broadcast` as defined in `api/intrinsics.py`.

    Args:
      value: As in `api/intrinsics.py`.

    Returns:
      As in `api/intrinsics.py`.

    Raises:
      TypeError: As in `api/intrinsics.py`.
    """
    value = value_impl.to_value(value, None, self._context_stack)
    type_utils.check_federated_value_placement(value, placements.SERVER,
                                               'value to be broadcasted')

    if not value.type_signature.all_equal:
      raise TypeError('The broadcasted value should be equal at all locations.')

    # TODO(b/113112108): Replace this hand-crafted logic here and below with
    # a call to a helper function that handles it in a uniform manner after
    # implementing support for correctly typechecking federated template types
    # and instantiating template types on concrete arguments.
    result_type = computation_types.FederatedType(value.type_signature.member,
                                                  placements.CLIENTS, True)
    intrinsic = value_impl.ValueImpl(
        computation_building_blocks.Intrinsic(
            intrinsic_defs.FEDERATED_BROADCAST.uri,
            computation_types.FunctionType(value.type_signature, result_type)),
        self._context_stack)
    return intrinsic(value)

  def federated_collect(self, value):
    """Implements `federated_collect` as defined in `api/intrinsics.py`.

    Args:
      value: As in `api/intrinsics.py`.

    Returns:
      As in `api/intrinsics.py`.

    Raises:
      TypeError: As in `api/intrinsics.py`.
    """
    value = value_impl.to_value(value, None, self._context_stack)
    type_utils.check_federated_value_placement(value, placements.CLIENTS,
                                               'value to be collected')

    result_type = computation_types.FederatedType(
        computation_types.SequenceType(value.type_signature.member),
        placements.SERVER, True)
    intrinsic = value_impl.ValueImpl(
        computation_building_blocks.Intrinsic(
            intrinsic_defs.FEDERATED_COLLECT.uri,
            computation_types.FunctionType(value.type_signature, result_type)),
        self._context_stack)
    return intrinsic(value)

  def federated_map(self, value, mapping_fn):
    """Implements `federated_map` as defined in `api/intrinsics.py`.

    Args:
      value: As in `api/intrinsics.py`.
      mapping_fn: As in `api/intrinsics.py`.

    Returns:
      As in `api/intrinsics.py`.

    Raises:
      TypeError: As in `api/intrinsics.py`.
    """

    # TODO(b/113112108): Possibly lift the restriction that the mapped value
    # must be placed at the clients after adding support for placement labels
    # in the federated types, and expanding the type specification of the
    # intrinsic this is based on to work with federated values of arbitrary
    # placement.

    value = value_impl.to_value(value, None, self._context_stack)
    if isinstance(value.type_signature, computation_types.NamedTupleType):
      if len(anonymous_tuple.to_elements(value.type_signature)) >= 2:
        # We've been passed a value which the user expects to be zipped.
        value = self.federated_zip(value)
    type_utils.check_federated_value_placement(value, placements.CLIENTS,
                                               'value to be mapped')

    # TODO(b/113112108): Add support for polymorphic templates auto-instantiated
    # here based on the actual type of the argument.
    mapping_fn = value_impl.to_value(mapping_fn, None, self._context_stack)

    py_typecheck.check_type(mapping_fn, value_base.Value)
    py_typecheck.check_type(mapping_fn.type_signature,
                            computation_types.FunctionType)
    if not type_utils.is_assignable_from(
        mapping_fn.type_signature.parameter, value.type_signature.member):
      raise TypeError(
          'The mapping function expects a parameter of type {}, but member '
          'constituents of the mapped value are of incompatible type {}.'
          .format(
              str(mapping_fn.type_signature.parameter_type),
              str(value.type_signature.member)))

    # TODO(b/113112108): Replace this as noted above.
    result_type = computation_types.FederatedType(
        mapping_fn.type_signature.result, placements.CLIENTS,
        value.type_signature.all_equal)
    intrinsic = value_impl.ValueImpl(
        computation_building_blocks.Intrinsic(
            intrinsic_defs.FEDERATED_MAP.uri,
            computation_types.FunctionType(value.type_signature, result_type)),
        self._context_stack)
    return intrinsic(value)

  def federated_reduce(self, value, zero, op):
    """Implements `federated_reduce` as defined in `api/intrinsics.py`.

    Args:
      value: As in `api/intrinsics.py`.
      zero: As in `api/intrinsics.py`.
      op: As in `api/intrinsics.py`.

    Returns:
      As in `api/intrinsics.py`.

    Raises:
      TypeError: As in `api/intrinsics.py`.
    """
    # TODO(b/113112108): Since in most cases, it can be assumed that CLIENTS is
    # a non-empty collective (or else, the computation fails), specifying zero
    # at this level of the API should probably be optional. TBD.

    value = value_impl.to_value(value, None, self._context_stack)
    type_utils.check_federated_value_placement(value, placements.CLIENTS,
                                               'value to be reduced')

    zero = value_impl.to_value(zero, None, self._context_stack)
    py_typecheck.check_type(zero, value_base.Value)

    # TODO(b/113112108): We need a check here that zero does not have federated
    # constituents.

    op = value_impl.to_value(op, None, self._context_stack)
    py_typecheck.check_type(op, value_base.Value)
    py_typecheck.check_type(op.type_signature, computation_types.FunctionType)
    op_type_expected = type_constructors.reduction_op(
        zero.type_signature, value.type_signature.member)
    if not type_utils.is_assignable_from(op_type_expected, op.type_signature):
      raise TypeError('Expected an operator of type {}, got {}.'.format(
          str(op_type_expected), str(op.type_signature)))

    # TODO(b/113112108): Replace this as noted above.
    result_type = computation_types.FederatedType(zero.type_signature,
                                                  placements.SERVER, True)
    intrinsic = value_impl.ValueImpl(
        computation_building_blocks.Intrinsic(
            intrinsic_defs.FEDERATED_REDUCE.uri,
            computation_types.FunctionType(
                [value.type_signature, zero.type_signature, op_type_expected],
                result_type)), self._context_stack)
    return intrinsic(value, zero, op)

  def federated_sum(self, value):
    """Implements `federated_sum` as defined in `api/intrinsics.py`.

    Args:
      value: As in `api/intrinsics.py`.

    Returns:
      As in `api/intrinsics.py`.

    Raises:
      TypeError: As in `api/intrinsics.py`.
    """
    value = value_impl.to_value(value, None, self._context_stack)
    type_utils.check_federated_value_placement(value, placements.CLIENTS,
                                               'value to be summed')

    if not type_utils.is_sum_compatible(value.type_signature):
      raise TypeError(
          'The value type {} is not compatible with the sum operator.'.format(
              str(value.type_signature)))

    # TODO(b/113112108): Replace this as noted above.
    result_type = computation_types.FederatedType(value.type_signature.member,
                                                  placements.SERVER, True)
    intrinsic = value_impl.ValueImpl(
        computation_building_blocks.Intrinsic(
            intrinsic_defs.FEDERATED_SUM.uri,
            computation_types.FunctionType(value.type_signature, result_type)),
        self._context_stack)
    return intrinsic(value)

  def federated_zip(self, value):
    """Implements `federated_zip` as defined in `api/intrinsics.py`.

    Args:
      value: As in `api/intrinsics.py`.

    Returns:
      As in `api/intrinsics.py`.

    Raises:
      TypeError: As in `api/intrinsics.py`.
    """
    # TODO(b/113112108): Extend this to accept *args.

    # TODO(b/113112108): Allow for auto-extraction of NamedTuples of length 1.

    # TODO(b/113112108): We use the iterate/unwrap approach below because
    # our type system is not powerful enough to express the concept of
    # "an operation that takes tuples of T of arbitrary length", and therefore
    # the intrinsic federated_zip must only take a fixed number of arguments,
    # here fixed at 2. There are other potential approaches to getting around
    # this problem (e.g. having the operator act on sequences and thereby
    # sidestepping the issue) which we may want to explore.

    value = value_impl.to_value(value, None, self._context_stack)
    py_typecheck.check_type(value, value_base.Value)
    py_typecheck.check_type(value.type_signature,
                            computation_types.NamedTupleType)
    num_elements = len(anonymous_tuple.to_elements(value.type_signature))
    if num_elements < 2:
      raise TypeError(
          'The federated zip operator zips tuples of at least two elements, '
          'but the tuple given as argument has {} '
          'elements.'.format(num_elements))
    for _, elem in anonymous_tuple.to_elements(value.type_signature):
      py_typecheck.check_type(elem, computation_types.FederatedType)
      if elem.placement is not placements.CLIENTS:
        raise TypeError(
            'The elements of the named tuple to zip must be placed at CLIENTS.')
    zipped = value_utils.zip_two_tuple(
        value_impl.to_value([value[0], value[1]], None, self._context_stack),
        self._context_stack)
    inputs = value_impl.to_value(
        computation_building_blocks.Reference(
            'inputs', zipped.type_signature.member), None, self._context_stack)
    flatten_func = value_impl.to_value(
        computation_building_blocks.Lambda(
            'inputs', zipped.type_signature.member,
            value_impl.ValueImpl.get_comp(inputs)), None, self._context_stack)
    for k in range(2, num_elements):
      zipped = value_utils.zip_two_tuple(
          value_impl.to_value([zipped, value[k]], None, self._context_stack),
          self._context_stack)
      # elements returns list of 2-tuples of the form (name, type)--the [-1][1]
      # grabs the type from the last element
      new_type = computation_types.to_type(
          anonymous_tuple.to_elements(zipped.type_signature.member)[-1][1])
      flatten_func = value_utils.flatten_first_index(flatten_func, new_type,
                                                     self._context_stack)
    return self.federated_map(zipped, flatten_func)

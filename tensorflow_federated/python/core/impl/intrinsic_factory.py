# Lint as: python3
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

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import value_base
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl import value_impl
from tensorflow_federated.python.core.impl import value_utils
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import type_factory
from tensorflow_federated.python.core.impl.context_stack import context_stack_base


class IntrinsicFactory(object):
  """A factory that can constructs intrinsics over the given context stack."""

  # Note: Tests for this class currently go to `api/intrinsics_test.py`.
  # For more documentation on the specific behaviors of each intrinsic, see
  # `api/intrinsics.py`.

  def __init__(self, context_stack):
    """Constructs this factory over the given context stack.

    Args:
      context_stack: The context stack to use.
    """
    py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
    self._context_stack = context_stack

  def federated_aggregate(self, value, zero, accumulate, merge, report):
    """Implements `federated_aggregate` as defined in `api/intrinsics.py`."""
    value = value_impl.to_value(value, None, self._context_stack)
    value = value_utils.ensure_federated_value(value, placements.CLIENTS,
                                               'value to be aggregated')

    zero = value_impl.to_value(zero, None, self._context_stack)
    py_typecheck.check_type(zero, value_base.Value)
    accumulate = value_impl.to_value(accumulate, None, self._context_stack)
    merge = value_impl.to_value(merge, None, self._context_stack)
    report = value_impl.to_value(report, None, self._context_stack)
    for op in [accumulate, merge, report]:
      py_typecheck.check_type(op, value_base.Value)
      py_typecheck.check_type(op.type_signature, computation_types.FunctionType)

    if not type_utils.is_assignable_from(accumulate.type_signature.parameter[0],
                                         zero.type_signature):
      raise TypeError('Expected `zero` to be assignable to type {}, '
                      'but was of incompatible type {}.'.format(
                          accumulate.type_signature.parameter[0],
                          zero.type_signature))

    accumulate_type_expected = type_factory.reduction_op(
        accumulate.type_signature.result, value.type_signature.member)
    merge_type_expected = type_factory.reduction_op(
        accumulate.type_signature.result, accumulate.type_signature.result)
    report_type_expected = computation_types.FunctionType(
        merge.type_signature.result, report.type_signature.result)
    for op_name, op, type_expected in [
        ('accumulate', accumulate, accumulate_type_expected),
        ('merge', merge, merge_type_expected),
        ('report', report, report_type_expected)
    ]:
      if not type_utils.is_assignable_from(type_expected, op.type_signature):
        raise TypeError(
            'Expected parameter `{}` to be of type {}, but received {} instead.'
            .format(op_name, type_expected, op.type_signature))

    value = value_impl.ValueImpl.get_comp(value)
    zero = value_impl.ValueImpl.get_comp(zero)
    accumulate = value_impl.ValueImpl.get_comp(accumulate)
    merge = value_impl.ValueImpl.get_comp(merge)
    report = value_impl.ValueImpl.get_comp(report)

    comp = building_block_factory.create_federated_aggregate(
        value, zero, accumulate, merge, report)
    return value_impl.ValueImpl(comp, self._context_stack)

  def federated_broadcast(self, value):
    """Implements `federated_broadcast` as defined in `api/intrinsics.py`."""
    value = value_impl.to_value(value, None, self._context_stack)
    value = value_utils.ensure_federated_value(value, placements.SERVER,
                                               'value to be broadcasted')

    if not value.type_signature.all_equal:
      raise TypeError('The broadcasted value should be equal at all locations.')

    value = value_impl.ValueImpl.get_comp(value)
    comp = building_block_factory.create_federated_broadcast(value)
    return value_impl.ValueImpl(comp, self._context_stack)

  def federated_collect(self, value):
    """Implements `federated_collect` as defined in `api/intrinsics.py`."""
    value = value_impl.to_value(value, None, self._context_stack)
    value = value_utils.ensure_federated_value(value, placements.CLIENTS,
                                               'value to be collected')

    value = value_impl.ValueImpl.get_comp(value)
    comp = building_block_factory.create_federated_collect(value)
    return value_impl.ValueImpl(comp, self._context_stack)

  def federated_eval(self, fn, placement):
    """Implements `federated_eval` as defined in `api/intrinsics.py`."""
    # TODO(b/113112108): Verify that neither the value, nor any of its parts
    # are of a federated type.

    fn = value_impl.to_value(fn, None, self._context_stack)
    py_typecheck.check_type(fn, value_base.Value)
    py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)

    if fn.type_signature.parameter is not None:
      raise TypeError(
          '`federated_eval` expects a `fn` that accepts no arguments, but '
          'the `fn` provided has a parameter of type {}.'.format(
              fn.type_signature.parameter))

    fn_comp = value_impl.ValueImpl.get_comp(fn)
    comp = building_block_factory.create_federated_eval(fn_comp, placement)
    return value_impl.ValueImpl(comp, self._context_stack)

  def federated_map(self, fn, arg):
    """Implements `federated_map` as defined in `api/intrinsics.py`."""
    # TODO(b/113112108): Possibly lift the restriction that the mapped value
    # must be placed at the server or clients. Would occur after adding support
    # for placement labels in the federated types, and expanding the type
    # specification of the intrinsic this is based on to work with federated
    # values of arbitrary placement.

    arg = value_impl.to_value(arg, None, self._context_stack)
    arg = value_utils.ensure_federated_value(arg, label='value to be mapped')

    # TODO(b/113112108): Add support for polymorphic templates auto-instantiated
    # here based on the actual type of the argument.
    fn = value_impl.to_value(fn, None, self._context_stack)

    py_typecheck.check_type(fn, value_base.Value)
    py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
    if not type_utils.is_assignable_from(fn.type_signature.parameter,
                                         arg.type_signature.member):
      raise TypeError(
          'The mapping function expects a parameter of type {}, but member '
          'constituents of the mapped value are of incompatible type {}.'
          .format(fn.type_signature.parameter, arg.type_signature.member))

    # TODO(b/144384398): Change structure to one that maps the placement type
    # to the building_block function that fits it, in a way that allows the
    # appropriate type checks.
    if arg.type_signature.placement is placements.SERVER:
      if not arg.type_signature.all_equal:
        raise TypeError(
            'Arguments placed at {} should be equal at all locations.'.format(
                placements.SERVER))
      fn = value_impl.ValueImpl.get_comp(fn)
      arg = value_impl.ValueImpl.get_comp(arg)
      comp = building_block_factory.create_federated_apply(fn, arg)
    elif arg.type_signature.placement is placements.CLIENTS:
      fn = value_impl.ValueImpl.get_comp(fn)
      arg = value_impl.ValueImpl.get_comp(arg)
      comp = building_block_factory.create_federated_map(fn, arg)
    else:
      raise TypeError(
          'The argument should be placed at {} or {}, placed at {} instead.'
          .format(placements.SERVER, placements.CLIENTS,
                  arg.type_signature.placement))

    return value_impl.ValueImpl(comp, self._context_stack)

  def federated_map_all_equal(self, fn, arg):
    """`federated_map` with the `all_equal` bit set in the `arg` and return."""
    # TODO(b/113112108): Possibly lift the restriction that the mapped value
    # must be placed at the clients after adding support for placement labels
    # in the federated types, and expanding the type specification of the
    # intrinsic this is based on to work with federated values of arbitrary
    # placement.
    arg = value_impl.to_value(arg, None, self._context_stack)
    arg = value_utils.ensure_federated_value(arg, placements.CLIENTS,
                                             'value to be mapped')

    # TODO(b/113112108): Add support for polymorphic templates auto-instantiated
    # here based on the actual type of the argument.
    fn = value_impl.to_value(fn, None, self._context_stack)

    py_typecheck.check_type(fn, value_base.Value)
    py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
    if not type_utils.is_assignable_from(fn.type_signature.parameter,
                                         arg.type_signature.member):
      raise TypeError(
          'The mapping function expects a parameter of type {}, but member '
          'constituents of the mapped value are of incompatible type {}.'
          .format(fn.type_signature.parameter, arg.type_signature.member))

    fn = value_impl.ValueImpl.get_comp(fn)
    arg = value_impl.ValueImpl.get_comp(arg)
    comp = building_block_factory.create_federated_map_all_equal(fn, arg)
    return value_impl.ValueImpl(comp, self._context_stack)

  def federated_mean(self, value, weight):
    """Implements `federated_mean` as defined in `api/intrinsics.py`."""
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
    value = value_utils.ensure_federated_value(value, placements.CLIENTS,
                                               'value to be averaged')
    if not type_utils.is_average_compatible(value.type_signature):
      raise TypeError(
          'The value type {} is not compatible with the average operator.'
          .format(value.type_signature))

    if weight is not None:
      weight = value_impl.to_value(weight, None, self._context_stack)
      weight = value_utils.ensure_federated_value(weight, placements.CLIENTS,
                                                  'weight to use in averaging')
      py_typecheck.check_type(weight.type_signature.member,
                              computation_types.TensorType)
      if weight.type_signature.member.shape.ndims != 0:
        raise TypeError('The weight type {} is not a federated scalar.'.format(
            weight.type_signature))
      if not (weight.type_signature.member.dtype.is_integer or
              weight.type_signature.member.dtype.is_floating):
        raise TypeError(
            'The weight type {} is not a federated integer or floating-point '
            'tensor.'.format(weight.type_signature))

    value = value_impl.ValueImpl.get_comp(value)
    if weight is not None:
      weight = value_impl.ValueImpl.get_comp(weight)
    comp = building_block_factory.create_federated_mean(value, weight)
    return value_impl.ValueImpl(comp, self._context_stack)

  def federated_reduce(self, value, zero, op):
    """Implements `federated_reduce` as defined in `api/intrinsics.py`."""
    # TODO(b/113112108): Since in most cases, it can be assumed that CLIENTS is
    # a non-empty collective (or else, the computation fails), specifying zero
    # at this level of the API should probably be optional. TBD.

    value = value_impl.to_value(value, None, self._context_stack)
    value = value_utils.ensure_federated_value(value, placements.CLIENTS,
                                               'value to be reduced')

    zero = value_impl.to_value(zero, None, self._context_stack)
    py_typecheck.check_type(zero, value_base.Value)

    # TODO(b/113112108): We need a check here that zero does not have federated
    # constituents.

    op = value_impl.to_value(op, None, self._context_stack)
    py_typecheck.check_type(op, value_base.Value)
    py_typecheck.check_type(op.type_signature, computation_types.FunctionType)
    op_type_expected = type_factory.reduction_op(zero.type_signature,
                                                 value.type_signature.member)
    if not type_utils.is_assignable_from(op_type_expected, op.type_signature):
      raise TypeError('Expected an operator of type {}, got {}.'.format(
          op_type_expected, op.type_signature))

    value = value_impl.ValueImpl.get_comp(value)
    zero = value_impl.ValueImpl.get_comp(zero)
    op = value_impl.ValueImpl.get_comp(op)
    comp = building_block_factory.create_federated_reduce(value, zero, op)
    return value_impl.ValueImpl(comp, self._context_stack)

  def federated_sum(self, value):
    """Implements `federated_sum` as defined in `api/intrinsics.py`."""
    value = value_impl.to_value(value, None, self._context_stack)
    value = value_utils.ensure_federated_value(value, placements.CLIENTS,
                                               'value to be summed')
    type_utils.check_is_sum_compatible(value.type_signature)
    value = value_impl.ValueImpl.get_comp(value)
    comp = building_block_factory.create_federated_sum(value)
    return value_impl.ValueImpl(comp, self._context_stack)

  def federated_value(self, value, placement):
    """Implements `federated_value` as defined in `api/intrinsics.py`."""
    # TODO(b/113112108): Verify that neither the value, nor any of its parts
    # are of a federated type.

    value = value_impl.to_value(value, None, self._context_stack)

    value = value_impl.ValueImpl.get_comp(value)
    comp = building_block_factory.create_federated_value(value, placement)
    return value_impl.ValueImpl(comp, self._context_stack)

  def federated_zip(self, value):
    """Implements `federated_zip` as defined in `api/intrinsics.py`."""
    # TODO(b/113112108): Extend this to accept *args.

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

    value = value_impl.ValueImpl.get_comp(value)
    comp = building_block_factory.create_federated_zip(value)
    return value_impl.ValueImpl(comp, self._context_stack)

  def federated_secure_sum(self, value, bitwidth):
    """Implements `federated_secure_sum` as defined in `api/intrinsics.py`."""
    value = value_impl.to_value(value, None, self._context_stack)
    value = value_utils.ensure_federated_value(value, placements.CLIENTS,
                                               'value to be summed')
    type_utils.check_is_structure_of_integers(value.type_signature)
    bitwidth = value_impl.to_value(bitwidth, None, self._context_stack)
    type_utils.check_equivalent_types(value.type_signature.member,
                                      bitwidth.type_signature)
    value = value_impl.ValueImpl.get_comp(value)
    bitwidth = value_impl.ValueImpl.get_comp(bitwidth)
    comp = building_block_factory.create_federated_secure_sum(value, bitwidth)
    return value_impl.ValueImpl(comp, self._context_stack)

  def sequence_map(self, fn, arg):
    """Implements `sequence_map` as defined in `api/intrinsics.py`."""
    fn = value_impl.to_value(fn, None, self._context_stack)
    py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
    arg = value_impl.to_value(arg, None, self._context_stack)

    if isinstance(arg.type_signature, computation_types.SequenceType):
      fn = value_impl.ValueImpl.get_comp(fn)
      arg = value_impl.ValueImpl.get_comp(arg)
      return value_impl.ValueImpl(
          building_block_factory.create_sequence_map(fn, arg),
          self._context_stack)
    elif isinstance(arg.type_signature, computation_types.FederatedType):
      parameter_type = computation_types.SequenceType(
          fn.type_signature.parameter)
      result_type = computation_types.SequenceType(fn.type_signature.result)
      intrinsic_type = computation_types.FunctionType(
          (fn.type_signature, parameter_type), result_type)
      intrinsic = building_blocks.Intrinsic(intrinsic_defs.SEQUENCE_MAP.uri,
                                            intrinsic_type)
      intrinsic_impl = value_impl.ValueImpl(intrinsic, self._context_stack)
      local_fn = value_utils.get_curried(intrinsic_impl)(fn)

      if arg.type_signature.placement in [
          placements.SERVER, placements.CLIENTS
      ]:
        return self.federated_map(local_fn, arg)
      else:
        raise TypeError('Unsupported placement {}.'.format(
            arg.type_signature.placement))
    else:
      raise TypeError(
          'Cannot apply `tff.sequence_map()` to a value of type {}.'.format(
              arg.type_signature))

  def sequence_reduce(self, value, zero, op):
    """Implements `sequence_reduce` as defined in `api/intrinsics.py`."""
    value = value_impl.to_value(value, None, self._context_stack)
    zero = value_impl.to_value(zero, None, self._context_stack)
    op = value_impl.to_value(op, None, self._context_stack)
    if isinstance(value.type_signature, computation_types.SequenceType):
      element_type = value.type_signature.element
    else:
      py_typecheck.check_type(value.type_signature,
                              computation_types.FederatedType)
      py_typecheck.check_type(value.type_signature.member,
                              computation_types.SequenceType)
      element_type = value.type_signature.member.element
    op_type_expected = type_factory.reduction_op(zero.type_signature,
                                                 element_type)
    if not type_utils.is_assignable_from(op_type_expected, op.type_signature):
      raise TypeError('Expected an operator of type {}, got {}.'.format(
          op_type_expected, op.type_signature))

    value = value_impl.ValueImpl.get_comp(value)
    zero = value_impl.ValueImpl.get_comp(zero)
    op = value_impl.ValueImpl.get_comp(op)
    if isinstance(value.type_signature, computation_types.SequenceType):
      return value_impl.ValueImpl(
          building_block_factory.create_sequence_reduce(value, zero, op),
          self._context_stack)
    else:
      value_type = computation_types.SequenceType(element_type)
      intrinsic_type = computation_types.FunctionType((
          value_type,
          zero.type_signature,
          op.type_signature,
      ), op.type_signature.result)
      intrinsic = building_blocks.Intrinsic(intrinsic_defs.SEQUENCE_REDUCE.uri,
                                            intrinsic_type)
      ref = building_blocks.Reference('arg', value_type)
      tup = building_blocks.Tuple((ref, zero, op))
      call = building_blocks.Call(intrinsic, tup)
      fn = building_blocks.Lambda(ref.name, ref.type_signature, call)
      fn_impl = value_impl.ValueImpl(fn, self._context_stack)
      if value.type_signature.placement in [
          placements.SERVER, placements.CLIENTS
      ]:
        return self.federated_map(fn_impl, value)
      else:
        raise TypeError('Unsupported placement {}.'.format(
            value.type_signature.placement))

  def sequence_sum(self, value):
    """Implements `sequence_sum` as defined in `api/intrinsics.py`."""
    value = value_impl.to_value(value, None, self._context_stack)
    if isinstance(value.type_signature, computation_types.SequenceType):
      element_type = value.type_signature.element
    else:
      py_typecheck.check_type(value.type_signature,
                              computation_types.FederatedType)
      py_typecheck.check_type(value.type_signature.member,
                              computation_types.SequenceType)
      element_type = value.type_signature.member.element
    type_utils.check_is_sum_compatible(element_type)

    if isinstance(value.type_signature, computation_types.SequenceType):
      value = value_impl.ValueImpl.get_comp(value)
      return value_impl.ValueImpl(
          building_block_factory.create_sequence_sum(value),
          self._context_stack)
    elif isinstance(value.type_signature, computation_types.FederatedType):
      intrinsic_type = computation_types.FunctionType(
          value.type_signature.member, value.type_signature.member.element)
      intrinsic = building_blocks.Intrinsic(intrinsic_defs.SEQUENCE_SUM.uri,
                                            intrinsic_type)
      intrinsic_impl = value_impl.ValueImpl(intrinsic, self._context_stack)
      if value.type_signature.placement in [
          placements.SERVER, placements.CLIENTS
      ]:
        return self.federated_map(intrinsic_impl, value)
      else:
        raise TypeError('Unsupported placement {}.'.format(
            value.type_signature.placement))
    else:
      raise TypeError(
          'Cannot apply `tff.sequence_sum()` to a value of type {}.'.format(
              value.type_signature))

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
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import value_base
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_base
from tensorflow_federated.python.core.impl.context_stack import symbol_binding_context
from tensorflow_federated.python.core.impl.federated_context import value_impl
from tensorflow_federated.python.core.impl.federated_context import value_utils
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_factory


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

  def _bind_comp_as_reference(self, comp):
    fc_context = self._context_stack.current
    if not isinstance(fc_context, symbol_binding_context.SymbolBindingContext):
      raise context_base.ContextError(
          'Intrsinics cannot be constructed without '
          'the ability to bind references to the '
          'generated ASTs; attempted to construct '
          'an intrinsic in context {c} which '
          'exposes no such mechanism.'.format(c=fc_context))
    return fc_context.bind_computation_to_reference(comp)

  def federated_aggregate(self, value, zero, accumulate, merge, report):
    """Implements `federated_aggregate` as defined in `api/intrinsics.py`."""
    value = value_impl.to_value(value, None, self._context_stack)
    value = value_utils.ensure_federated_value(value, placements.CLIENTS,
                                               'value to be aggregated')

    zero = value_impl.to_value(zero, None, self._context_stack)
    py_typecheck.check_type(zero, value_base.Value)
    accumulate = value_impl.to_value(
        accumulate,
        None,
        self._context_stack,
        parameter_type_hint=computation_types.StructType(
            [zero.type_signature, value.type_signature.member]))
    merge = value_impl.to_value(
        merge,
        None,
        self._context_stack,
        parameter_type_hint=computation_types.StructType(
            [zero.type_signature, zero.type_signature]))
    report = value_impl.to_value(
        report,
        None,
        self._context_stack,
        parameter_type_hint=zero.type_signature)
    for op in [accumulate, merge, report]:
      py_typecheck.check_type(op, value_base.Value)
      py_typecheck.check_type(op.type_signature, computation_types.FunctionType)

    if not accumulate.type_signature.parameter[0].is_assignable_from(
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
      if not type_expected.is_assignable_from(op.type_signature):
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
    comp = self._bind_comp_as_reference(comp)
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
    comp = self._bind_comp_as_reference(comp)
    return value_impl.ValueImpl(comp, self._context_stack)

  def federated_collect(self, value):
    """Implements `federated_collect` as defined in `api/intrinsics.py`."""
    value = value_impl.to_value(value, None, self._context_stack)
    value = value_utils.ensure_federated_value(value, placements.CLIENTS,
                                               'value to be collected')

    value = value_impl.ValueImpl.get_comp(value)
    comp = building_block_factory.create_federated_collect(value)
    comp = self._bind_comp_as_reference(comp)
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
    comp = self._bind_comp_as_reference(comp)
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

    fn = value_impl.to_value(
        fn,
        None,
        self._context_stack,
        parameter_type_hint=arg.type_signature.member)

    py_typecheck.check_type(fn, value_base.Value)
    py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
    if not fn.type_signature.parameter.is_assignable_from(
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
          'Expected `arg` to have a type with a supported placement, '
          'found {}.'.format(arg.type_signature.placement))

    comp = self._bind_comp_as_reference(comp)
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

    fn = value_impl.to_value(
        fn,
        None,
        self._context_stack,
        parameter_type_hint=arg.type_signature.member)

    py_typecheck.check_type(fn, value_base.Value)
    py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
    if not fn.type_signature.parameter.is_assignable_from(
        arg.type_signature.member):
      raise TypeError(
          'The mapping function expects a parameter of type {}, but member '
          'constituents of the mapped value are of incompatible type {}.'
          .format(fn.type_signature.parameter, arg.type_signature.member))

    fn = value_impl.ValueImpl.get_comp(fn)
    arg = value_impl.ValueImpl.get_comp(arg)
    comp = building_block_factory.create_federated_map_all_equal(fn, arg)
    comp = self._bind_comp_as_reference(comp)
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
    if not type_analysis.is_average_compatible(value.type_signature):
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
    comp = self._bind_comp_as_reference(comp)
    return value_impl.ValueImpl(comp, self._context_stack)

  def federated_sum(self, value):
    """Implements `federated_sum` as defined in `api/intrinsics.py`."""
    value = value_impl.to_value(value, None, self._context_stack)
    value = value_utils.ensure_federated_value(value, placements.CLIENTS,
                                               'value to be summed')
    type_analysis.check_is_sum_compatible(value.type_signature)
    value = value_impl.ValueImpl.get_comp(value)
    comp = building_block_factory.create_federated_sum(value)
    comp = self._bind_comp_as_reference(comp)
    return value_impl.ValueImpl(comp, self._context_stack)

  def federated_value(self, value, placement):
    """Implements `federated_value` as defined in `api/intrinsics.py`."""
    value = value_impl.to_value(value, None, self._context_stack)
    if type_analysis.contains(value.type_signature, lambda t: t.is_federated()):
      raise TypeError('Cannot place value {} containing federated types at '
                      'another placement; requested to be placed at {}.'.format(
                          value, placement))

    value = value_impl.ValueImpl.get_comp(value)
    comp = building_block_factory.create_federated_value(value, placement)
    comp = self._bind_comp_as_reference(comp)
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
    py_typecheck.check_type(value.type_signature, computation_types.StructType)

    value = value_impl.ValueImpl.get_comp(value)
    comp = building_block_factory.create_federated_zip(value)
    comp = self._bind_comp_as_reference(comp)
    return value_impl.ValueImpl(comp, self._context_stack)

  def federated_secure_sum(self, value, bitwidth):
    """Implements `federated_secure_sum` as defined in `api/intrinsics.py`."""
    value = value_impl.to_value(value, None, self._context_stack)
    value = value_utils.ensure_federated_value(value, placements.CLIENTS,
                                               'value to be summed')
    type_analysis.check_is_structure_of_integers(value.type_signature)
    bitwidth_value = value_impl.to_value(bitwidth, None, self._context_stack)
    value_member_type = value.type_signature.member
    bitwidth_type = bitwidth_value.type_signature
    if not type_analysis.is_valid_bitwidth_type_for_value_type(
        bitwidth_type, value_member_type):
      raise TypeError(
          'Expected `federated_secure_sum` parameter `bitwidth` to match '
          'the structure of `value`, with one integer bitwidth per tensor in '
          '`value`. Found `value` of `{}` and `bitwidth` of `{}`.'.format(
              value_member_type, bitwidth_type))
    if bitwidth_type.is_tensor() and value_member_type.is_struct():
      bitwidth_value = value_impl.to_value(
          structure.map_structure(lambda _: bitwidth, value_member_type), None,
          self._context_stack)
    value = value_impl.ValueImpl.get_comp(value)
    bitwidth_value = value_impl.ValueImpl.get_comp(bitwidth_value)
    comp = building_block_factory.create_federated_secure_sum(
        value, bitwidth_value)
    comp = self._bind_comp_as_reference(comp)
    return value_impl.ValueImpl(comp, self._context_stack)

  def sequence_map(self, fn, arg):
    """Implements `sequence_map` as defined in `api/intrinsics.py`."""
    fn = value_impl.to_value(fn, None, self._context_stack)
    py_typecheck.check_type(fn.type_signature, computation_types.FunctionType)
    arg = value_impl.to_value(arg, None, self._context_stack)

    if arg.type_signature.is_sequence():
      fn = value_impl.ValueImpl.get_comp(fn)
      arg = value_impl.ValueImpl.get_comp(arg)
      comp = building_block_factory.create_sequence_map(fn, arg)
      comp = self._bind_comp_as_reference(comp)
      return value_impl.ValueImpl(comp, self._context_stack)
    elif arg.type_signature.is_federated():
      parameter_type = computation_types.SequenceType(
          fn.type_signature.parameter)
      result_type = computation_types.SequenceType(fn.type_signature.result)
      intrinsic_type = computation_types.FunctionType(
          (fn.type_signature, parameter_type), result_type)
      intrinsic = building_blocks.Intrinsic(intrinsic_defs.SEQUENCE_MAP.uri,
                                            intrinsic_type)
      intrinsic_impl = value_impl.ValueImpl(intrinsic, self._context_stack)
      local_fn = value_utils.get_curried(intrinsic_impl)(fn)
      return self.federated_map(local_fn, arg)
    else:
      raise TypeError(
          'Cannot apply `tff.sequence_map()` to a value of type {}.'.format(
              arg.type_signature))

  def sequence_reduce(self, value, zero, op):
    """Implements `sequence_reduce` as defined in `api/intrinsics.py`."""
    value = value_impl.to_value(value, None, self._context_stack)
    zero = value_impl.to_value(zero, None, self._context_stack)
    op = value_impl.to_value(op, None, self._context_stack)
    # Check if the value is a federated sequence that should be reduced
    # under a `federated_map`.
    if value.type_signature.is_federated():
      is_federated_sequence = True
      value_member_type = value.type_signature.member
      value_member_type.check_sequence()
      zero_member_type = zero.type_signature.member
    else:
      is_federated_sequence = False
      value.type_signature.check_sequence()
    value = value_impl.ValueImpl.get_comp(value)
    zero = value_impl.ValueImpl.get_comp(zero)
    op = value_impl.ValueImpl.get_comp(op)
    if not is_federated_sequence:
      comp = building_block_factory.create_sequence_reduce(value, zero, op)
      comp = self._bind_comp_as_reference(comp)
      return value_impl.ValueImpl(comp, self._context_stack)
    else:
      ref_type = computation_types.StructType(
          [value_member_type, zero_member_type])
      ref = building_blocks.Reference('arg', ref_type)
      arg1 = building_blocks.Selection(ref, index=0)
      arg2 = building_blocks.Selection(ref, index=1)
      call = building_block_factory.create_sequence_reduce(arg1, arg2, op)
      fn = building_blocks.Lambda(ref.name, ref.type_signature, call)
      fn_value_impl = value_impl.ValueImpl(fn, self._context_stack)
      args = building_blocks.Struct([value, zero])
      return self.federated_map(fn_value_impl, args)

  def sequence_sum(self, value):
    """Implements `sequence_sum` as defined in `api/intrinsics.py`."""
    value = value_impl.to_value(value, None, self._context_stack)
    if value.type_signature.is_sequence():
      element_type = value.type_signature.element
    else:
      py_typecheck.check_type(value.type_signature,
                              computation_types.FederatedType)
      py_typecheck.check_type(value.type_signature.member,
                              computation_types.SequenceType)
      element_type = value.type_signature.member.element
    type_analysis.check_is_sum_compatible(element_type)

    if value.type_signature.is_sequence():
      value = value_impl.ValueImpl.get_comp(value)
      comp = building_block_factory.create_sequence_sum(value)
      comp = self._bind_comp_as_reference(comp)
      return value_impl.ValueImpl(comp, self._context_stack)
    elif value.type_signature.is_federated():
      intrinsic_type = computation_types.FunctionType(
          value.type_signature.member, value.type_signature.member.element)
      intrinsic = building_blocks.Intrinsic(intrinsic_defs.SEQUENCE_SUM.uri,
                                            intrinsic_type)
      intrinsic_impl = value_impl.ValueImpl(intrinsic, self._context_stack)
      return self.federated_map(intrinsic_impl, value)
    else:
      raise TypeError(
          'Cannot apply `tff.sequence_sum()` to a value of type {}.'.format(
              value.type_signature))

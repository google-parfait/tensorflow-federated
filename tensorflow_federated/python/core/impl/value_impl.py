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
"""Implementations of the abstract interface Value in api/value_base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import attr
import six
from six.moves import range
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import value_base
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import placement_literals
from tensorflow_federated.python.core.impl import tensorflow_serialization
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.utils import dtype_utils
from tensorflow_federated.python.core.impl.utils import function_utils
from tensorflow_federated.python.core.impl.utils import graph_utils


@six.add_metaclass(abc.ABCMeta)
class ValueImpl(value_base.Value):
  """A generic base class for values that appear in TFF computations."""

  def __init__(self, comp, context_stack):
    """Constructs a value of the given type.

    Args:
      comp: An instance of building_blocks.ComputationBuildingBlock
        that contains the logic that computes this value.
      context_stack: The context stack to use.
    """
    py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
    py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
    super(ValueImpl, self).__setattr__('_comp', comp)
    super(ValueImpl, self).__setattr__('_context_stack', context_stack)

  @property
  def type_signature(self):
    return self._comp.type_signature

  @classmethod
  def get_comp(cls, value):
    py_typecheck.check_type(value, cls)
    return value._comp  # pylint: disable=protected-access

  @classmethod
  def get_context_stack(cls, value):
    py_typecheck.check_type(value, cls)
    return value._context_stack  # pylint: disable=protected-access

  def __repr__(self):
    return repr(self._comp)

  def __str__(self):
    return str(self._comp)

  def __dir__(self):
    if not isinstance(self._comp.type_signature,
                      computation_types.NamedTupleType):
      raise TypeError(
          'Operator dir() is only suppored for named tuples, but the object '
          'on which it has been invoked is of type {}.'.format(
              str(self._comp.type_signature)))
    return dir(self._comp.type_signature)

  def __getattr__(self, name):
    py_typecheck.check_type(name, six.string_types)
    if (isinstance(self._comp.type_signature, computation_types.FederatedType)
        and isinstance(self._comp.type_signature.member,
                       computation_types.NamedTupleType)):
      return ValueImpl(
          building_block_factory.create_federated_getattr_call(
              self._comp, name), self._context_stack)
    elif not isinstance(self._comp.type_signature,
                        computation_types.NamedTupleType):
      raise TypeError(
          'Operator getattr() is only supported for named tuples, but the '
          'object on which it has been invoked is of type {}.'.format(
              str(self._comp.type_signature)))
    if name not in dir(self._comp.type_signature):
      raise AttributeError(
          'There is no such attribute as \'{}\' in this tuple.'.format(name))
    if isinstance(self._comp, building_blocks.Tuple):
      return ValueImpl(getattr(self._comp, name), self._context_stack)
    return ValueImpl(
        building_blocks.Selection(self._comp, name=name), self._context_stack)

  def __setattr__(self, name, value):
    py_typecheck.check_type(name, six.string_types)
    value_comp = ValueImpl.get_comp(to_value(value, None, self._context_stack))
    if isinstance(self._comp.type_signature,
                  computation_types.FederatedType) and isinstance(
                      self._comp.type_signature.member,
                      computation_types.NamedTupleType):
      new_comp = building_block_factory.create_federated_setattr_call(
          self._comp, name, value_comp)
      super(ValueImpl, self).__setattr__('_comp', new_comp)
      return
    elif not isinstance(self._comp.type_signature,
                        computation_types.NamedTupleType):
      raise TypeError(
          'Operator setattr() is only supported for named tuples, but the '
          'object on which it has been invoked is of type {}.'.format(
              str(self._comp.type_signature)))
    named_tuple_setattr_lambda = building_block_factory.create_named_tuple_setattr_lambda(
        self._comp.type_signature, name, value_comp)
    new_comp = building_blocks.Call(named_tuple_setattr_lambda, self._comp)
    super(ValueImpl, self).__setattr__('_comp', new_comp)

  def __len__(self):
    type_signature = self._comp.type_signature
    if isinstance(type_signature, computation_types.FederatedType):
      type_signature = type_signature.member
    if not isinstance(type_signature, computation_types.NamedTupleType):
      raise TypeError(
          'Operator len() is only supported for (possibly federated) named '
          'tuples, but the object on which it has been invoked is of type '
          '{}.'.format(str(self._comp.type_signature)))
    return len(type_signature)

  def __getitem__(self, key):
    py_typecheck.check_type(key, (int, slice))
    if (isinstance(self._comp.type_signature, computation_types.FederatedType)
        and isinstance(self._comp.type_signature.member,
                       computation_types.NamedTupleType)):
      return ValueImpl(
          building_block_factory.create_federated_getitem_call(self._comp, key),
          self._context_stack)
    if not isinstance(self._comp.type_signature,
                      computation_types.NamedTupleType):
      raise TypeError(
          'Operator getitem() is only supported for named tuples, but the '
          'object on which it has been invoked is of type {}.'.format(
              str(self._comp.type_signature)))
    elem_length = len(self._comp.type_signature)
    if isinstance(key, int):
      if key < 0 or key >= elem_length:
        raise IndexError(
            'The index of the selected element {} is out of range.'.format(key))
      if isinstance(self._comp, building_blocks.Tuple):
        return ValueImpl(self._comp[key], self._context_stack)
      else:
        return ValueImpl(
            building_blocks.Selection(self._comp, index=key),
            self._context_stack)
    elif isinstance(key, slice):
      index_range = range(*key.indices(elem_length))
      if not index_range:
        raise IndexError('Attempted to slice 0 elements, which is not '
                         'currently supported.')
      return to_value([self[k] for k in index_range], None, self._context_stack)

  def __iter__(self):
    type_signature = self._comp.type_signature
    if isinstance(type_signature, computation_types.FederatedType):
      type_signature = type_signature.member
    if not isinstance(type_signature, computation_types.NamedTupleType):
      raise TypeError(
          'Operator iter() is only supported for (possibly federated) named '
          'tuples, but the object on which it has been invoked is of type '
          '{}.'.format(str(self._comp.type_signature)))
    for index in range(len(type_signature)):
      yield self[index]

  def __call__(self, *args, **kwargs):
    if not isinstance(self._comp.type_signature,
                      computation_types.FunctionType):
      raise SyntaxError(
          'Function-like invocation is only supported for values of '
          'functional types, but the value being invoked is of type '
          '{} that does not support invocation.'.format(
              str(self._comp.type_signature)))
    if args or kwargs:
      args = [to_value(x, None, self._context_stack) for x in args]
      kwargs = {
          k: to_value(v, None, self._context_stack)
          for k, v in six.iteritems(kwargs)
      }
      arg = function_utils.pack_args(self._comp.type_signature.parameter, args,
                                     kwargs, self._context_stack.current)
      arg = ValueImpl.get_comp(to_value(arg, None, self._context_stack))
    else:
      arg = None
    return ValueImpl(building_blocks.Call(self._comp, arg), self._context_stack)

  def __add__(self, other):
    other = to_value(other, None, self._context_stack)
    if not type_utils.are_equivalent_types(self.type_signature,
                                           other.type_signature):
      raise TypeError('Cannot add {} and {}.'.format(
          str(self.type_signature), str(other.type_signature)))
    return ValueImpl(
        building_blocks.Call(
            building_blocks.Intrinsic(
                intrinsic_defs.GENERIC_PLUS.uri,
                computation_types.FunctionType(
                    [self.type_signature, self.type_signature],
                    self.type_signature)),
            ValueImpl.get_comp(
                to_value([self, other], None, self._context_stack))),
        self._context_stack)


def _wrap_constant_as_value(const, context_stack):
  """Wraps the given Python constant as a `tff.Value`.

  Args:
    const: Python constant to be converted to TFF value. Anything convertible to
      Tensor via `tf.constant` can be passed in.
    context_stack: The context stack to use.

  Returns:
    An instance of `value_base.Value`.
  """
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  tf_comp, _ = tensorflow_serialization.serialize_py_fn_as_tf_computation(
      lambda: tf.constant(const), None, context_stack)
  compiled_comp = building_blocks.CompiledComputation(tf_comp)
  called_comp = building_blocks.Call(compiled_comp)
  return ValueImpl(called_comp, context_stack)


def _wrap_sequence_as_value(elements, element_type, context_stack):
  """Wraps `elements` as a TFF sequence with elements of type `element_type`.

  Args:
    elements: Python object to the wrapped as a TFF sequence value.
    element_type: An instance of `Type` that determines the type of elements of
      the sequence.
    context_stack: The context stack to use.

  Returns:
    An instance of `tff.Value`.

  Raises:
    TypeError: If `elements` and `element_type` are of incompatible types.
  """
  # TODO(b/113116813): Add support for other representations of sequences.
  py_typecheck.check_type(elements, list)
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)

  # Checks that the types of all the individual elements are compatible with the
  # requested type of the sequence as a while.
  for elem in elements:
    elem_type = type_utils.infer_type(elem)
    if not type_utils.is_assignable_from(element_type, elem_type):
      raise TypeError(
          'Expected all sequence elements to be {}, found {}.'.format(
              str(element_type), str(elem_type)))

  # Defines a no-arg function that builds a `tf.data.Dataset` from the elements.
  def _create_dataset_from_elements():
    return graph_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), elements, element_type)

  # Wraps the dataset as a value backed by a no-argument TensorFlow computation.
  tf_comp, _ = tensorflow_serialization.serialize_py_fn_as_tf_computation(
      _create_dataset_from_elements, None, context_stack)
  return ValueImpl(
      building_blocks.Call(building_blocks.CompiledComputation(tf_comp)),
      context_stack)


def to_value(arg, type_spec, context_stack):
  """Converts the argument into an instance of `tff.Value`.

  The types of non-`tff.Value` arguments that are currently convertible to
  `tff.Value` include the following:

  * Lists, tuples, anonymous tuples, named tuples, and dictionaries, all
    of which are converted into instances of `tff.Tuple`.
  * Placement literals, converted into instances of `tff.Placement`.
  * Computations.
  * Python constants of type `str`, `int`, `float`, `bool`
  * Numpy objects inherting from `np.ndarray` or `np.generic` (the parent
    of numpy scalar types)

  Args:
    arg: Either an instance of `tff.Value`, or an argument convertible to
      `tff.Value`. The argument must not be `None`.
    type_spec: A type specifier that allows for disambiguating the target type
      (e.g., when two TFF types can be mapped to the same Python
      representations), or `None` if none available, in which case TFF tries to
      determine the type of the TFF value automatically.
    context_stack: The context stack to use.

  Returns:
    An instance of `tff.Value` corresponding to the given `arg`, and of TFF type
    matching the `type_spec` if specified (not `None`).

  Raises:
    TypeError: if `arg` is of an unsupported type, or of a type that does not
      match `type_spec`. Raises explicit error message if TensorFlow constructs
      are encountered, as TensorFlow code should be sealed away from TFF
      federated context.
  """
  py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
  if type_spec is not None:
    type_spec = computation_types.to_type(type_spec)
    type_utils.check_well_formed(type_spec)
  if isinstance(arg, ValueImpl):
    result = arg
  elif isinstance(arg, building_blocks.ComputationBuildingBlock):
    result = ValueImpl(arg, context_stack)
  elif isinstance(arg, placement_literals.PlacementLiteral):
    result = ValueImpl(building_blocks.Placement(arg), context_stack)
  elif isinstance(arg, computation_base.Computation):
    result = ValueImpl(
        building_blocks.CompiledComputation(
            computation_impl.ComputationImpl.get_proto(arg)), context_stack)
  elif type_spec is not None and isinstance(type_spec,
                                            computation_types.SequenceType):
    result = _wrap_sequence_as_value(arg, type_spec.element, context_stack)
  elif isinstance(arg, anonymous_tuple.AnonymousTuple):
    result = ValueImpl(
        building_blocks.Tuple([
            (k, ValueImpl.get_comp(to_value(v, None, context_stack)))
            for k, v in anonymous_tuple.to_elements(arg)
        ]), context_stack)
  elif py_typecheck.is_named_tuple(arg):
    result = to_value(arg._asdict(), None, context_stack)
  elif py_typecheck.is_attrs(arg):
    result = to_value(
        attr.asdict(arg, dict_factory=collections.OrderedDict, recurse=False),
        None, context_stack)
  elif isinstance(arg, dict):
    if isinstance(arg, collections.OrderedDict):
      items = six.iteritems(arg)
    else:
      items = sorted(six.iteritems(arg))
    value = building_blocks.Tuple([
        (k, ValueImpl.get_comp(to_value(v, None, context_stack)))
        for k, v in items
    ])
    result = ValueImpl(value, context_stack)
  elif isinstance(arg, (tuple, list)):
    result = ValueImpl(
        building_blocks.Tuple([
            ValueImpl.get_comp(to_value(x, None, context_stack)) for x in arg
        ]), context_stack)
  elif isinstance(arg, dtype_utils.TENSOR_REPRESENTATION_TYPES):
    result = _wrap_constant_as_value(arg, context_stack)
  elif isinstance(arg, (tf.Tensor, tf.Variable)):
    raise TypeError(
        'TensorFlow construct {} has been encountered in a federated '
        'context. TFF does not support mixing TF and federated orchestration '
        'code. Please wrap any TensorFlow constructs with '
        '`tff.tf_computation`.'.format(arg))
  else:
    raise TypeError(
        'Unable to interpret an argument of type {} as a TFF value.'.format(
            py_typecheck.type_string(type(arg))))
  py_typecheck.check_type(result, ValueImpl)
  if (type_spec is not None and
      not type_utils.is_assignable_from(type_spec, result.type_signature)):
    raise TypeError(
        'The supplied argument maps to TFF type {}, which is incompatible '
        'with the requested type {}.'.format(
            str(result.type_signature), str(type_spec)))
  return result

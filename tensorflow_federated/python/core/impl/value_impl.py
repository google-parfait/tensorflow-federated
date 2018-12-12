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
# Dependency imports
import numpy as np
import six
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import types
from tensorflow_federated.python.core.api import value_base

from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import func_utils
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import placement_literals
from tensorflow_federated.python.core.impl import tensorflow_serialization


@six.add_metaclass(abc.ABCMeta)
class ValueImpl(value_base.Value):
  """A generic base class for values that appear in TFF computations."""

  def __init__(self, comp):
    """Constructs a value of the given type.

    Args:
      comp: An instance of computation_building_blocks.ComputationBuildingBlock
        that contains the logic that computes this value.
    """
    py_typecheck.check_type(
        comp, computation_building_blocks.ComputationBuildingBlock)
    self._comp = comp

  @property
  def type_signature(self):
    return self._comp.type_signature

  @classmethod
  def get_comp(cls, value):
    py_typecheck.check_type(value, cls)
    return value._comp  # pylint: disable=protected-access

  def __repr__(self):
    return repr(self._comp)

  def __str__(self):
    return str(self._comp)

  def __dir__(self):
    if not isinstance(self._comp.type_signature, types.NamedTupleType):
      raise TypeError(
          'Operator dir() is only suppored for named tuples, but the object '
          'on which it has been invoked is of type {}.'.format(
              str(self._comp.type_signature)))
    else:
      # Not pre-creating or memoizing the list, as we do not expect this to be
      # a common enough operation to warrant doing so.
      return [e[0] for e in self._comp.type_signature.elements if e[0]]

  def __getattr__(self, name):
    py_typecheck.check_type(name, six.string_types)
    if not isinstance(self._comp.type_signature, types.NamedTupleType):
      raise TypeError(
          'Operator getattr() is only supported for named tuples, but the '
          'object on which it has been invoked is of type {}.'.format(
              str(self._comp.type_signature)))
    if name not in [x for x, _ in self._comp.type_signature.elements]:
      raise AttributeError(
          'There is no such attribute as \'{}\' in this tuple.'.format(name))
    if isinstance(self._comp, computation_building_blocks.Tuple):
      return ValueImpl(getattr(self._comp, name))
    else:
      return ValueImpl(
          computation_building_blocks.Selection(self._comp, name=name))

  def __len__(self):
    if not isinstance(self._comp.type_signature, types.NamedTupleType):
      raise TypeError(
          'Operator len() is only supported for named tuples, but the object '
          'on which it has been invoked is of type {}.'.format(
              str(self._comp.type_signature)))
    else:
      return len(self._comp.type_signature.elements)

  def __getitem__(self, key):
    py_typecheck.check_type(key, (int, slice))
    if not isinstance(self._comp.type_signature, types.NamedTupleType):
      raise TypeError(
          'Operator getitem() is only supported for named tuples, but the '
          'object on which it has been invoked is of type {}.'.format(
              str(self._comp.type_signature)))
    elem_length = len(self._comp.type_signature.elements)
    if isinstance(key, int):
      if key < 0 or key >= elem_length:
        raise IndexError(
            'The index of the selected element {} is out of range.'.format(key))
      if isinstance(self._comp, computation_building_blocks.Tuple):
        return ValueImpl(self._comp[key])
      else:
        return ValueImpl(
            computation_building_blocks.Selection(self._comp, index=key))
    elif isinstance(key, slice):
      index_range = range(*key.indices(elem_length))
      if not index_range:
        raise IndexError('Attempted to slice 0 elements, which is not '
                         'currently supported.')
      return to_value([self[k] for k in index_range])

  def __iter__(self):
    if not isinstance(self._comp.type_signature, types.NamedTupleType):
      raise TypeError(
          'Operator iter() is only supported for named tuples, but the object '
          'on which it has been invoked is of type {}.'.format(
              str(self._comp.type_signature)))
    else:
      for index in range(len(self._comp.type_signature.elements)):
        yield self[index]

  def __call__(self, *args, **kwargs):
    if not isinstance(self._comp.type_signature, types.FunctionType):
      raise SyntaxError(
          'Function-like invocation is only supported for values of '
          'functional types, but the value being invoked is of type '
          '{} that does not support invocation.'.format(
              str(self._comp.type_signature)))
    else:
      if args or kwargs:
        args = [to_value(x) for x in args]
        kwargs = {k: to_value(v) for k, v in six.iteritems(kwargs)}
        arg = func_utils.pack_args(self._comp.type_signature.parameter, args,
                                   kwargs)
        arg = ValueImpl.get_comp(to_value(arg))
      else:
        arg = None
      return ValueImpl(computation_building_blocks.Call(self._comp, arg))

  def __add__(self, other):
    other = to_value(other)
    if (not self.type_signature.is_assignable_from(other.type_signature) or
        not other.type_signature.is_assignable_from(self.type_signature)):
      raise TypeError('Cannot add {} and {}.'.format(
          str(self.type_signature), str(other.type_signature)))
    return ValueImpl(
        computation_building_blocks.Call(
            computation_building_blocks.Intrinsic(
                intrinsic_defs.GENERIC_PLUS.uri,
                types.FunctionType([self.type_signature, self.type_signature],
                                   self.type_signature)),
            to_value([self, other])))


def _wrap_constant_as_value(const):
  """Wraps the given Python constant as a TFF `Value`.

  Args:
    const: Python constant to be converted to TFF value. Allowable types are
      `str`, `int`, `float`, `boolean`, or `numpy.ndarray`.

  Returns:
    Instance of `value_base.Value`.
  """
  if not isinstance(const, (str, int, float, bool, np.ndarray)):
    raise TypeError('Please pass one of str, int, float, bool, or '
                    'numpy ndarray to value_impl._wrap_constant_as_value')
  lam = lambda: tf.constant(const)
  tf_comp = tensorflow_serialization.serialize_py_func_as_tf_computation(lam)
  compiled_comp = computation_building_blocks.CompiledComputation(tf_comp)
  called_comp = computation_building_blocks.Call(compiled_comp)
  return ValueImpl(called_comp)


def to_value(arg):
  """Converts the argument into an instance of Value.

  Args:
    arg: Either an instance of Value, or an argument convertible to Value. The
      argument must not be None. The types of non-Value arguments that are
      currently convertible to Value include the following:  * Lists, tuples,
        anonymous tuples, named tuples, and dictionaries, all of which are
        converted into instances of Tuple. * Placement literals, converted into
        instances of `Placement`. * Computations. * Python constants of type
        str, int, float, bool, or numpy ndarray.

  Returns:
    An instance of Value corresponding to the given 'arg'.

  Raises:
    TypeError: if 'arg' is of an unsupported type.
  """
  if isinstance(arg, ValueImpl):
    return arg
  elif isinstance(arg, computation_building_blocks.ComputationBuildingBlock):
    return ValueImpl(arg)
  elif isinstance(arg, anonymous_tuple.AnonymousTuple):
    return ValueImpl(
        computation_building_blocks.Tuple([(k, ValueImpl.get_comp(
            to_value(v))) for k, v in anonymous_tuple.to_elements(arg)]))
  elif '_asdict' in vars(type(arg)):
    return to_value(arg._asdict())
  elif isinstance(arg, dict):
    return ValueImpl(
        computation_building_blocks.Tuple([
            (k, ValueImpl.get_comp(to_value(v))) for k, v in six.iteritems(arg)
        ]))
  elif isinstance(arg, (tuple, list)):
    return ValueImpl(
        computation_building_blocks.Tuple(
            [ValueImpl.get_comp(to_value(x)) for x in arg]))
  elif isinstance(arg, placement_literals.PlacementLiteral):
    return ValueImpl(computation_building_blocks.Placement(arg))
  elif isinstance(arg, computation_base.Computation):
    return ValueImpl(
        computation_building_blocks.CompiledComputation(
            computation_impl.ComputationImpl.get_proto(arg)))
  elif isinstance(arg, (str, int, float, bool, np.ndarray)):
    return _wrap_constant_as_value(arg)
  else:
    raise TypeError(
        'Unable to interpret an argument of type {} as a TFF value.'.format(
            py_typecheck.type_string(type(arg))))

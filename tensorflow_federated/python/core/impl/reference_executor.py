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
"""A simple interpreted reference executor.

This executor is designed for simplicity, not for performance. It is intended
for use in unit tests, as the golden standard and point of comparison for other
executors. Unit test suites for other executors should include a test that runs
them side by side and compares their results against this executor for a number
of computations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

# Dependency imports

import six
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import graph_utils
from tensorflow_federated.python.core.impl import tensorflow_deserialization
from tensorflow_federated.python.core.impl import transformations


@six.add_metaclass(abc.ABCMeta)
class ComputedValue(object):
  """A base class for all kinds of values computed by the reference executor."""

  def __init__(self, type_spec):
    """Constructs a value with the given type spec.

    Args:
      type_spec: Type signature or something convertible to it.
    """
    type_spec = computation_types.to_type(type_spec)
    py_typecheck.check_type(type_spec, computation_types.Type)
    self._type_signature = type_spec

  @property
  def type_signature(self):
    return self._type_signature

  @abc.abstractproperty
  def value(self):
    """Returns a Python objectm, the form of which is a function of TFF type."""
    raise NotImplementedError


class TensorValue(ComputedValue):
  """A class to represent tensor values computed by the executor."""

  def __init__(self, value, type_spec):
    """Constructs a computed tensor value.

    Args:
      value: An instance of one of the types used to represent computed values
        of tensors, such as a numpy array.
      type_spec: The type signature.
    """
    type_spec = computation_types.to_type(type_spec)
    py_typecheck.check_type(type_spec, computation_types.TensorType)
    super(TensorValue, self).__init__(type_spec)

    # TODO(b/113123634): This is where we might verify that `value` is a kind
    # of value that would be computed from a tensor (a numpy or simple type).

    self._value = value

  @property
  def value(self):
    return self._value


class NamedTupleValue(ComputedValue):
  """A class to represent named tuple values computed by the executor."""

  def __init__(self, value_tuple, type_spec):
    """Constructs a computed named tuple value.

    Args:
      value_tuple: An instance of `AnonymousTuple` with _ComputedValue members.
      type_spec: The type signature.
    """
    type_spec = computation_types.to_type(type_spec)
    py_typecheck.check_type(type_spec, computation_types.NamedTupleType)
    py_typecheck.check_type(value_tuple, anonymous_tuple.AnonymousTuple)
    for e in value_tuple:
      py_typecheck.check_type(e, ComputedValue)
    value_elements = anonymous_tuple.to_elements(value_tuple)
    if len(value_elements) != len(type_spec.elements):
      raise ValueError(
          'The number of elements {} in the value tuple {} does not match the '
          'type spec {}.'.format(
              len(value_elements), str(value_tuple), str(type_spec)))
    for index, (elem_name, elem_type) in enumerate(type_spec.elements):
      value_name, value = value_elements[index]
      if value_name != elem_name:
        raise ValueError(
            'Found value name {}, where {} was expected at position {} '
            'in the value tuple.'.format(value_name, elem_name, index))
      if not elem_type.is_assignable_from(value.type_signature):
        raise ValueError(
            'Found value of type {}, where {} was expected at position {} '
            'in the value tuple.'.format(
                str(value.type_signature), str(elem_type), index))
    super(NamedTupleValue, self).__init__(type_spec)
    self._value = value_tuple

  @property
  def value(self):
    return self._value


class FunctionValue(ComputedValue):
  """A class to represent function values computed by the executor."""

  def __init__(self, function, type_spec):
    """Constructs a computed function value.

    Args:
      function: A one-parameter Python function that expects a `ComputedValue`
        argument, and returns a `ComputedValue` result.
      type_spec: The type signature of this function.
    """
    type_spec = computation_types.to_type(type_spec)
    py_typecheck.check_type(type_spec, computation_types.FunctionType)
    super(FunctionValue, self).__init__(type_spec)
    self._value = function

  @property
  def value(self):
    return self._value


def stamp_computed_value_into_graph(value, graph):
  """Stamps `value` in `graph`.

  Args:
    value: An instance of `ComputedValue`.
    graph: The graph to stamp in.

  Returns:
    A Python object made of tensors staped into `graph`, `tf.data.Dataset`s,
    and `AnonymousTuple`s that structurally corresponds to the value passed
    at input.
  """
  if value is None:
    return None
  else:
    py_typecheck.check_type(value, ComputedValue)
    py_typecheck.check_type(graph, tf.Graph)
    if isinstance(value, TensorValue):
      with graph.as_default():
        return tf.constant(
            value.value,
            dtype=value.type_signature.dtype,
            shape=value.type_signature.shape)
    elif isinstance(value, NamedTupleValue):
      elements = anonymous_tuple.to_elements(value.value)
      return anonymous_tuple.AnonymousTuple(
          [(k, stamp_computed_value_into_graph(v, graph)) for k, v in elements])
    else:
      raise NotImplementedError(
          'Unable to embed a computed value of type {} in graph.'.format(
              str(value.type_signature)))


def to_computed_value(value, type_spec):
  """Creates a `ComputedValue` from the raw `value` and given `type_spec`.

  Args:
    value: The payload.
    type_spec: The TFF type.

  Returns:
    An instance of `ComputedValue` matching the payload and type.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)
  if isinstance(type_spec, computation_types.TensorType):
    return TensorValue(value, type_spec)
  elif isinstance(type_spec, computation_types.NamedTupleType):
    py_typecheck.check_type(value, anonymous_tuple.AnonymousTuple)
    result_elements = []
    for index, (elem_name, elem_type) in enumerate(type_spec.elements):
      result_elements.append((elem_name,
                              to_computed_value(value[index], elem_type)))
    return NamedTupleValue(
        anonymous_tuple.AnonymousTuple(result_elements), type_spec)
  else:
    # TODO(b/113123634): Add support for data sets.
    raise NotImplementedError(
        'Unable to construct a computed value for type {}.'.format(
            str(type_spec)))


def to_raw_value(value):
  """Returns a raw value embedded in `ComputedValue`.

  Args:
    value: An instance of `ComputedValue`.

  Returns:
    The raw value extracted from `value`.
  """
  py_typecheck.check_type(value, ComputedValue)
  if isinstance(value, TensorValue):
    return value.value
  elif isinstance(value, NamedTupleValue):
    return anonymous_tuple.AnonymousTuple(
        [(k, to_raw_value(v))
         for k, v in anonymous_tuple.to_elements(value.value)])
  else:
    # TODO(b/113123634): Add support for the remaining types of values.
    raise NotImplementedError(
        'Unable to extract a raw value from computed value of type {}.'.format(
            py_typecheck.type_string(type(value))))


class ReferenceExecutor(executor_base.Executor):
  """A simple interpreted reference executor.

  This executor is to be used by default in unit tests and simple applications
  such as colab notebooks and turorials. It is intended to serve as the gold
  standard of correctness for all other executors to compare against. As such,
  it is designed for simplicity and ease of reasoning about correctness, rather
  than for high performance. We will tolerate copying values, marshaling and
  unmarshaling when crossing TF graph boundary, etc., for the sake of keeping
  the logic minimal. The executor can be reused across multiple calls, so any
  state associated with individual executions is maintained separately from
  this class. High-performance simulations on large data sets will require a
  separate executor optimized for performance. This executor is plugged in as
  the handler of computation invocations at the top level of the context stack.
  """

  def __init__(self):
    """Creates a reference executor."""

    # TODO(b/113116813): Add a way to declare environmental bindings here,
    # e.g., a way to specify how data URIs are mapped to physical resources.
    pass

  def execute(self, computation_proto):
    """Runs the given self-contained computation, and returns the final results.

    Args:
      computation_proto: An instance of `Computation` proto to execute. It must
        be self-contained, i.e., it must not declare any parameters (all inputs
        it needs should be baked into its body by the time it is submitted for
        execution by the default execution context that spawns it).

    Returns:
      The result produced by the computation (the format to be described).

    Raises:
      NotImplementedError: At the moment, every time when invoked.
    """
    py_typecheck.check_type(computation_proto, pb.Computation)
    comp = computation_building_blocks.ComputationBuildingBlock.from_proto(
        computation_proto)
    if (isinstance(comp.type_signature, computation_types.FunctionType) and
        comp.type_signature.parameter is not None):
      raise ValueError(
          'Computations submitted for execution must not declare any '
          'parameters, but the executor found a parameter of type {}.'.format(
              str(comp.type_signature.parameter)))
    comp = transformations.name_compiled_computations(comp)
    return to_raw_value(self._compute(comp))

  def _compute(self, comp):
    """Computes `comp` and returns the resulting computed value.

    Args:
      comp: An instance of `ComputationBuildingBlock`.

    Returns:
      The corresponding instance of `ComputedValue` that represents the result
      of `comp`.
    """
    if isinstance(comp, computation_building_blocks.CompiledComputation):
      computation_oneof = comp.proto.WhichOneof('computation')
      if computation_oneof != 'tensorflow':
        raise ValueError(
            'Expected all parsed compiled computations to be tensorflow, '
            'but found \'{}\' instead.'.format(computation_oneof))
      else:
        return FunctionValue(lambda x: self._run_tensorflow(comp, x),
                             comp.type_signature)
    elif isinstance(comp, computation_building_blocks.Call):
      computed_func = self._compute(comp.function)
      py_typecheck.check_type(computed_func, FunctionValue)
      if comp.argument is not None:
        computed_arg = self._compute(comp.argument)
      else:
        computed_arg = None
      return computed_func.value(computed_arg)
    elif isinstance(comp, computation_building_blocks.Tuple):
      value_tuple = anonymous_tuple.AnonymousTuple(
          [(k, self._compute(v)) for k, v in anonymous_tuple.to_elements(comp)])
      return NamedTupleValue(value_tuple, comp.type_signature)
    else:
      raise NotImplementedError(
          'A computation building block of a type not currently recognized '
          'by the reference executor: {}.'.format(str(comp)))

  def _run_tensorflow(self, comp, arg):
    """Runs a compiled TensorFlow computation `comp` with argument `arg`.

    Args:
      comp: An instance of `CompiledComputation` with embedded TensorFlow code.
      arg: An instance of `ComputedValue` that represents the argument, or None
        if the compuation expects no argument.

    Returns:
      An instance of `ComputedValue` with the result.
    """
    with tf.Graph().as_default() as graph:
      stamped_arg = stamp_computed_value_into_graph(arg, graph)
      result = tensorflow_deserialization.deserialize_and_call_tf_computation(
          comp.proto, stamped_arg, graph)
      with tf.Session(graph=graph) as sess:
        result_val = graph_utils.fetch_value_in_session(result, sess)
    return to_computed_value(result_val, comp.type_signature.result)

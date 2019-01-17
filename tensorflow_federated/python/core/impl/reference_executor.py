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

# Dependency imports
import numpy as np
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
from tensorflow_federated.python.core.impl import type_utils


class ComputedValue(object):
  """A container for values computed by the reference executor."""

  def __init__(self, value, type_spec):
    """Creates a value with given raw payload `value` and TFF type `type_spec`.

    For performance reasons, the constructor does not check that the payload is
    of the corresponding type. It is the responsibility of the caller to do so,
    e.g., by calling the helper function `check_representation_matches_type()`.
    See the definition of this function for the value representations.

    Args:
      value: The raw payload (the representation of the computed value), the
        exact form of which depends on the type, as describd above.
      type_spec: An instance of `tff.Type` or something convertible to it that
        describes the TFF type of this value.
    """
    type_spec = computation_types.to_type(type_spec)
    py_typecheck.check_type(type_spec, computation_types.Type)
    self._type_signature = type_spec
    self._value = value

  @property
  def type_signature(self):
    return self._type_signature

  @property
  def value(self):
    return self._value


# TODO(b/113123634): Address this in a more systematic way, and possibly narrow
# this down to a smaller set.
_TENSOR_REPRESENTATION_TYPES = (str, int, float, bool, np.int32, np.int64,
                                np.float32, np.float64, np.ndarray)


def check_representation_matches_type(value, type_spec):
  """Checks that payload (value representation) `value` matches `type_spec`.

    The accepted forms of payload as as follows:

    * For TFF tensor types, either primitive Python types such as `str`, `int`,
      `float`, and `bool`, Numpy primitive types (such as `np.int32`), and
      Numpy arrays (`np.ndarray`), as listed in `_TENSOR_REPRESENTATION_TYPES`.

    * For TFF named tuple types, instances of `anonymous_tuple.AnonymousTuple`.

    * For TFF functional types, Python callables that accept a single argument
      that is a `ComputedValue` (if the function has a parameter) or `None`
      (otherwise), and return a `ComputedValue` in the result.

  Args:
    value: The raw representation of the value to verify against `type_spec`.
    type_spec: The TFF type, an instance of `tff.Type` or something convertible
      to it.

  Raises:
    TypeError: If `value` is not a valid representation for given `type_spec`.
    NotImplementedError: If verification for `type_spec` is not supported.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)

  # NOTE: We do not simply call `type_utils.infer_type()` on `value`, as the
  # representations of values in the refernece executor are only a subset of
  # the Python types recognized by that helper function.

  if isinstance(type_spec, computation_types.TensorType):
    py_typecheck.check_type(value, _TENSOR_REPRESENTATION_TYPES)
    inferred_type_spec = type_utils.infer_type(value)
    if inferred_type_spec != type_spec:
      raise TypeError(
          'The tensor type {} of the value representation does not match '
          'the type spec {}.'.format(str(inferred_type_spec), str(type_spec)))
  elif isinstance(type_spec, computation_types.NamedTupleType):
    py_typecheck.check_type(value, anonymous_tuple.AnonymousTuple)
    value_elements = anonymous_tuple.to_elements(value)
    type_spec_elements = anonymous_tuple.to_elements(type_spec)
    if len(value_elements) != len(type_spec_elements):
      raise TypeError(
          'The number of elements {} in the value tuple {} does not match the '
          'number of elements {} in the type spec {}.'.format(
              len(value_elements), str(value), len(type_spec_elements),
              str(type_spec)))
    for index, (type_elem_name, type_elem) in enumerate(type_spec_elements):
      value_elem_name, value_elem = value_elements[index]
      if value_elem_name != type_elem_name:
        raise TypeError(
            'Found element named {} where {} was expected at position {} '
            'in the value tuple.'.format(value_elem_name, type_elem_name,
                                         index))
      check_representation_matches_type(value_elem, type_elem)
  elif isinstance(type_spec, computation_types.FunctionType):
    py_typecheck.check_callable(value)
  else:
    raise NotImplementedError(
        'Unable to verify value representation of type {}.'.format(
            str(type_spec)))


def stamp_computed_value_into_graph(value, graph):
  """Stamps `value` in `graph`.

  Args:
    value: An instance of `ComputedValue`.
    graph: The graph to stamp in.

  Returns:
    A Python object made of tensors stamped into `graph`, `tf.data.Dataset`s,
    and `AnonymousTuple`s that structurally corresponds to the value passed
    at input.
  """
  if value is None:
    return None
  else:
    py_typecheck.check_type(value, ComputedValue)
    check_representation_matches_type(value.value, value.type_signature)
    py_typecheck.check_type(graph, tf.Graph)
    if isinstance(value.type_signature, computation_types.TensorType):
      with graph.as_default():
        return tf.constant(
            value.value,
            dtype=value.type_signature.dtype,
            shape=value.type_signature.shape)
    elif isinstance(value.type_signature, computation_types.NamedTupleType):
      elements = anonymous_tuple.to_elements(value.value)
      type_elements = anonymous_tuple.to_elements(value.type_signature)
      stamped_elements = []
      for idx, (k, v) in enumerate(elements):
        computed_v = ComputedValue(v, type_elements[idx][1])
        stamped_v = stamp_computed_value_into_graph(computed_v, graph)
        stamped_elements.append((k, stamped_v))
      return anonymous_tuple.AnonymousTuple(stamped_elements)
    else:
      # TODO(b/113123634): Add support for embedding sequences (`tf.Dataset`s).

      raise NotImplementedError(
          'Unable to embed a computed value of type {} in graph.'.format(
              str(value.type_signature)))


def capture_computed_value_from_graph(value, type_spec):
  """Captures `value` from a TensorFlow graph.

  Args:
    value: A Python object made of tensors in `graph`, `tf.data.Dataset`s,
      `AnonymousTuple`s and other structures, to be captured as an instance of
      `ComputedValue`.
    type_spec: The type of the value to be captured.

  Returns:
    An instance of `ComputedValue`.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)

  # TODO(b/113123634): Add handling for things like `tf.Dataset`s, as well as
  # possibly other Python structures that don't match the kinds of permitted
  # representations for pyaloads (see `check_representation_matches_type()`).

  check_representation_matches_type(value, type_spec)
  return ComputedValue(value, type_spec)


def run_tensorflow(comp, arg):
  """Runs a compiled TensorFlow computation `comp` with argument `arg`.

  Args:
    comp: An instance of `computation_building_blocks.CompiledComputation` with
      embedded TensorFlow code.
    arg: An instance of `ComputedValue` that represents the argument, or `None`
      if the compuation expects no argument.

  Returns:
    An instance of `ComputedValue` with the result.
  """
  py_typecheck.check_type(comp, computation_building_blocks.CompiledComputation)
  if arg is not None:
    py_typecheck.check_type(arg, ComputedValue)
  with tf.Graph().as_default() as graph:
    stamped_arg = stamp_computed_value_into_graph(arg, graph)
    result = tensorflow_deserialization.deserialize_and_call_tf_computation(
        comp.proto, stamped_arg, graph)
  with tf.Session(graph=graph) as sess:
    result_val = graph_utils.fetch_value_in_session(result, sess)
  return capture_computed_value_from_graph(result_val,
                                           comp.type_signature.result)


class ComputationContext(object):
  """Encapsulates context/state in which computations or parts thereof run."""

  def __init__(self, parent_context=None, local_symbols=None):
    """Constructs a new execution context.

    Args:
      parent_context: The parent context, or `None` if this is the root.
      local_symbols: The dictionary of local symbols defined in this context, or
        `None` if there are none. The keys (names) are of a string type, and the
        values (what the names bind to) are of type `ComputedValue`.
    """
    if parent_context is not None:
      py_typecheck.check_type(parent_context, ComputationContext)
    self._parent_context = parent_context
    self._local_symbols = {}
    if local_symbols is not None:
      py_typecheck.check_type(local_symbols, dict)
      for k, v in six.iteritems(local_symbols):
        py_typecheck.check_type(k, six.string_types)
        py_typecheck.check_type(v, ComputedValue)
        self._local_symbols[str(k)] = v

  def resolve_reference(self, name):
    """Resolves the given reference `name` in this context.

    Args:
      name: The string name to resolve.

    Returns:
      An instance of `ComputedValue` corresponding to this name.

    Raises:
      ValueError: If the name cannot be resolved.
    """
    py_typecheck.check_type(name, six.string_types)
    value = self._local_symbols.get(str(name))
    if value is not None:
      return value
    elif self._parent_context is not None:
      return self._parent_context.resolve_reference(name)
    else:
      raise ValueError(
          'The name \'{}\' is not defined in this context.'.format(name))


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
      ValueError: If the computation is malformed.
      TypeError: If type mismatch occurs anywhere in the course of computing.
    """
    py_typecheck.check_type(computation_proto, pb.Computation)
    comp = computation_building_blocks.ComputationBuildingBlock.from_proto(
        computation_proto)
    if isinstance(comp.type_signature, computation_types.FunctionType):
      if comp.type_signature.parameter is not None:
        raise ValueError(
            'Computations submitted for execution must not declare any '
            'parameters, but the executor found a parameter of '
            'type {}.'.format(str(comp.type_signature.parameter)))
      else:
        expected_result_type = comp.type_signature.result
    else:
      expected_result_type = comp.type_signature

    comp = transformations.name_compiled_computations(comp)
    result = self._compute(comp, ComputationContext())

    if not type_utils.is_assignable_from(expected_result_type,
                                         result.type_signature):
      raise TypeError(
          'The type {} of the result does not match the return type {} of '
          'the computation.'.format(
              str(result.type_signature), str(expected_result_type)))
    return result.value

  def _compute(self, comp, context):
    """Computes `comp` and returns the resulting computed value.

    Args:
      comp: An instance of
        `computation_building_blocks.ComputationBuildingBlock`.
      context: An instance of `ComputationContext`.

    Returns:
      The corresponding instance of `ComputedValue` that represents the result
      of `comp`.

    Raises:
      TypeError: If type mismatch occurs during the course of computation.
      ValueError: If a malformed value is encountered.
      NotImplementedError: For computation building blocks that are not yet
        supported by this executor.
    """
    if isinstance(comp, computation_building_blocks.CompiledComputation):
      return self._compute_compiled(comp, context)
    elif isinstance(comp, computation_building_blocks.Call):
      return self._compute_call(comp, context)
    elif isinstance(comp, computation_building_blocks.Tuple):
      return self._compute_tuple(comp, context)
    elif isinstance(comp, computation_building_blocks.Reference):
      return self._compute_reference(comp, context)
    elif isinstance(comp, computation_building_blocks.Selection):
      return self._compute_selection(comp, context)
    elif isinstance(comp, computation_building_blocks.Lambda):
      return self._compute_lambda(comp, context)
    elif isinstance(comp, computation_building_blocks.Block):
      return self._compute_block(comp, context)
    elif isinstance(comp, computation_building_blocks.Intrinsic):
      return self._compute_intrinsic(comp, context)
    elif isinstance(comp, computation_building_blocks.Data):
      return self._compute_data(comp, context)
    elif isinstance(comp, computation_building_blocks.Placement):
      return self._compute_placement(comp, context)
    else:
      raise NotImplementedError(
          'A computation building block of a type {} not currently recognized '
          'by the reference executor: {}.'.format(str(type(comp)), str(comp)))

  def _compute_compiled(self, comp, context):
    py_typecheck.check_type(comp,
                            computation_building_blocks.CompiledComputation)
    computation_oneof = comp.proto.WhichOneof('computation')
    if computation_oneof != 'tensorflow':
      raise ValueError(
          'Expected all parsed compiled computations to be tensorflow, '
          'but found \'{}\' instead.'.format(computation_oneof))
    else:
      return ComputedValue(lambda x: run_tensorflow(comp, x),
                           comp.type_signature)

  def _compute_call(self, comp, context):
    py_typecheck.check_type(comp, computation_building_blocks.Call)
    computed_func = self._compute(comp.function, context)
    py_typecheck.check_type(computed_func.type_signature,
                            computation_types.FunctionType)
    if comp.argument is not None:
      computed_arg = self._compute(comp.argument, context)
      if not type_utils.is_assignable_from(
          computed_func.type_signature.parameter, computed_arg.type_signature):
        raise TypeError(
            'The type {} of the argument does not match the '
            'type {} expected by the function being invoked.'.format(
                str(computed_arg.type_signature),
                str(computed_func.type_signature.parameter)))
    else:
      computed_arg = None
    result = computed_func.value(computed_arg)
    py_typecheck.check_type(result, ComputedValue)
    if not type_utils.is_assignable_from(computed_func.type_signature.result,
                                         result.type_signature):
      raise TypeError('The type {} of the result does not match the '
                      'type {} returned by the invoked function.'.format(
                          str(result.type_signature),
                          str(computed_func.type_signature.result)))
    return result

  def _compute_tuple(self, comp, context):
    py_typecheck.check_type(comp, computation_building_blocks.Tuple)
    result_elements = []
    result_type_elements = []
    for k, v in anonymous_tuple.to_elements(comp):
      computed_v = self._compute(v, context)
      if not type_utils.is_assignable_from(v.type_signature,
                                           computed_v.type_signature):
        raise TypeError(
            'The computed type {} of a tuple element does not match '
            'the declated type {}.'.format(
                str(computed_v.type_signature), str(v.type_signature)))
      result_elements.append((k, computed_v.value))
      result_type_elements.append((k, computed_v.type_signature))
    return ComputedValue(
        anonymous_tuple.AnonymousTuple(result_elements),
        computation_types.NamedTupleType(
            [(k, v) if k else v for k, v in result_type_elements]))

  def _compute_selection(self, comp, context):
    py_typecheck.check_type(comp, computation_building_blocks.Selection)
    source = self._compute(comp.source, context)
    py_typecheck.check_type(source.type_signature,
                            computation_types.NamedTupleType)
    py_typecheck.check_type(source.value, anonymous_tuple.AnonymousTuple)
    if comp.name is not None:
      result_value = getattr(source.value, comp.name)
      result_type = getattr(source.type_signature, comp.name)
    else:
      assert comp.index is not None
      result_value = source.value[comp.index]
      result_type = source.type_signature[comp.index]
    if not type_utils.is_assignable_from(comp.type_signature, result_type):
      raise TypeError(
          'Expected the result of selection to be {}, found {}.'.format(
              str(comp.type_signature), str(result_type)))
    return ComputedValue(result_value, result_type)

  def _compute_lambda(self, comp, context):
    py_typecheck.check_type(comp, computation_building_blocks.Lambda)
    py_typecheck.check_type(context, ComputationContext)

    def _wrap(arg):
      py_typecheck.check_type(arg, ComputedValue)
      if not type_utils.is_assignable_from(comp.parameter_type,
                                           arg.type_signature):
        raise TypeError(
            'Expected the type of argument {} to be {}, found {}.'.format(
                str(comp.parameter_name), str(comp.parameter_type),
                str(arg.type_signature)))
      return ComputationContext(context, {comp.parameter_name: arg})

    return ComputedValue(lambda x: self._compute(comp.result, _wrap(x)),
                         comp.type_signature)

  def _compute_reference(self, comp, context):
    py_typecheck.check_type(comp, computation_building_blocks.Reference)
    py_typecheck.check_type(context, ComputationContext)
    return context.resolve_reference(comp.name)

  def _compute_block(self, comp, context):
    py_typecheck.check_type(comp, computation_building_blocks.Block)
    py_typecheck.check_type(context, ComputationContext)
    for local_name, local_comp in comp.locals:
      local_val = self._compute(local_comp, context)
      context = ComputationContext(context, {local_name: local_val})
    return self._compute(comp.result, context)

  def _compute_intrinsic(self, comp, context):
    py_typecheck.check_type(comp, computation_building_blocks.Intrinsic)
    raise NotImplementedError('Intrinsic is currently unsupported.')

  def _compute_data(self, comp, context):
    py_typecheck.check_type(comp, computation_building_blocks.Data)
    raise NotImplementedError('Data is currently unsupported.')

  def _compute_placement(self, comp, context):
    py_typecheck.check_type(comp, computation_building_blocks.Placement)
    raise NotImplementedError('Placement is currently unsupported.')

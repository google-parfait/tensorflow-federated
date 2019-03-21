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

import collections

import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import compiler_pipeline
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import context_base
from tensorflow_federated.python.core.impl import dtype_utils
from tensorflow_federated.python.core.impl import graph_utils
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import placement_literals
from tensorflow_federated.python.core.impl import tensorflow_deserialization
from tensorflow_federated.python.core.impl import transformations
from tensorflow_federated.python.core.impl import type_constructors
from tensorflow_federated.python.core.impl import type_utils


class ComputedValue(object):
  """A container for values computed by the reference executor."""

  def __init__(self, value, type_spec):
    """Creates a value with given raw payload `value` and TFF type `type_spec`.

    For performance reasons, the constructor does not check that the payload is
    of the corresponding type. It is the responsibility of the caller to do so,
    e.g., by calling the helper function `to_representation_for_type()`.
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

  def __str__(self):
    return 'ComputedValue({}, {})'.format(
        str(self._value), str(self._type_signature))


def to_representation_for_type(value, type_spec, callable_handler=None):
  """Verifies or converts the `value` representation to match `type_spec`.

  This method first tries to determine whether `value` is a valid representation
  of TFF type `type_spec`. If so, it is returned unchanged. If not, but if it
  can be converted into a valid representation, it is converted to such, and the
  valid representation is returned. If no conversion to a valid representation
  is possible, TypeError is raised.

  The accepted forms of `value` for vaqrious TFF types as as follows:

  * For TFF tensor types listed in `dtypes.TENSOR_REPRESENTATION_TYPES`.

  * For TFF named tuple types, instances of `anonymous_tuple.AnonymousTuple`.

  * For TFF sequences, Python lists.

  * For TFF functional types, Python callables that accept a single argument
    that is an instance of `ComputedValue` (if the function has a parameter)
    or `None` (otherwise), and return a `ComputedValue` instance as a result.
    This function only verifies that `value` is a callable.

  * For TFF abstract types, there is no valid representation. The reference
    executor requires all types in an executable computation to be concrete.

  * For TFF placement types, the valid representations are the placement
    literals (currently only `tff.SERVER` and `tff.CLIENTS`).

  * For TFF federated types with `all_equal` set to `True`, the representation
    is the same as the representation of the member constituent (thus, e.g.,
    a valid representation of `int32@SERVER` is the same as that of `int32`).
    For those types that have `all_equal_` set to `False`, the representation
    is a Python list of member constituents.

    NOTE: This function does not attempt at validating that the sizes of lists
    that represent federated values match the corresponding placemenets. The
    cardinality analysis is a separate step, handled by the reference executor
    at a different point. As long as values can be packed into a Python list,
    they are accepted as they are.

  Args:
    value: The raw representation of a value to compare against `type_spec` and
      potentially to be converted into a canonical form for the given TFF type.
    type_spec: The TFF type, an instance of `tff.Type` or something convertible
      to it that determines what the valid representation should be.
    callable_handler: The function to invoke to handle TFF functional types. If
      this is `None`, functional types are not supported. The function must
      accept `value` and `type_spec` as arguments and return the converted valid
      representation, just as `to_representation_for_type`.

  Returns:
    Either `value` itself, or the `value` converted into a valid representation
    for `type_spec`.

  Raises:
    TypeError: If `value` is not a valid representation for given `type_spec`.
    NotImplementedError: If verification for `type_spec` is not supported.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)
  if callable_handler is not None:
    py_typecheck.check_callable(callable_handler)

  # NOTE: We do not simply call `type_utils.infer_type()` on `value`, as the
  # representations of values in the reference executor are only a subset of
  # the Python types recognized by that helper function.

  if isinstance(type_spec, computation_types.TensorType):
    if tf.executing_eagerly() and isinstance(value, tf.Tensor):
      value = value.numpy()
    py_typecheck.check_type(value, dtype_utils.TENSOR_REPRESENTATION_TYPES)
    inferred_type_spec = type_utils.infer_type(value)
    if not type_utils.is_assignable_from(type_spec, inferred_type_spec):
      raise TypeError(
          'The tensor type {} of the value representation does not match '
          'the type spec {}.'.format(str(inferred_type_spec), str(type_spec)))
    return value
  elif isinstance(type_spec, computation_types.NamedTupleType):
    type_spec_elements = anonymous_tuple.to_elements(type_spec)
    # Special-casing unodered dictionaries to allow their elements to be fed in
    # the order in which they're defined in the named tuple type.
    if (isinstance(value, dict) and
        (set(value.keys()) == set(k for k, _ in type_spec_elements))):
      value = collections.OrderedDict(
          [(k, value[k]) for k, _ in type_spec_elements])
    value = anonymous_tuple.from_container(value)
    value_elements = anonymous_tuple.to_elements(value)
    if len(value_elements) != len(type_spec_elements):
      raise TypeError(
          'The number of elements {} in the value tuple {} does not match the '
          'number of elements {} in the type spec {}.'.format(
              len(value_elements), str(value), len(type_spec_elements),
              str(type_spec)))
    result_elements = []
    for index, (type_elem_name, type_elem) in enumerate(type_spec_elements):
      value_elem_name, value_elem = value_elements[index]
      if value_elem_name not in [type_elem_name, None]:
        raise TypeError(
            'Found element named `{}` where `{}` was expected at position {} '
            'in the value tuple. Value: {}. Type: {}'.format(
                value_elem_name, type_elem_name, index, value, type_spec))
      converted_value_elem = to_representation_for_type(value_elem, type_elem,
                                                        callable_handler)
      result_elements.append((type_elem_name, converted_value_elem))
    return anonymous_tuple.AnonymousTuple(result_elements)
  elif isinstance(type_spec, computation_types.SequenceType):
    if isinstance(value, tf.data.Dataset):
      if tf.executing_eagerly():
        return [
            to_representation_for_type(v, type_spec.element, callable_handler)
            for v in value
        ]
      else:
        raise ValueError(
            'Processing `tf.data.Datasets` outside of eager mode is not '
            'currently supported.')
    return [
        to_representation_for_type(v, type_spec.element, callable_handler)
        for v in value
    ]
  elif isinstance(type_spec, computation_types.FunctionType):
    if callable_handler is not None:
      return callable_handler(value, type_spec)
    else:
      raise TypeError(
          'Values that are callables have been explicitly disallowed '
          'in this context. If you would like to supply here a function '
          'as a parameter, please construct a computation that contains '
          'this call.')
  elif isinstance(type_spec, computation_types.AbstractType):
    raise TypeError(
        'Abstract types are not supported by the reference executor.')
  elif isinstance(type_spec, computation_types.PlacementType):
    py_typecheck.check_type(value, placement_literals.PlacementLiteral)
    return value
  elif isinstance(type_spec, computation_types.FederatedType):
    if type_spec.all_equal:
      return to_representation_for_type(value, type_spec.member,
                                        callable_handler)
    elif type_spec.placement is not placements.CLIENTS:
      raise TypeError(
          'Unable to determine a valid value representation for a federated '
          'type with non-equal members placed at {}.'.format(
              str(type_spec.placement)))
    elif not isinstance(value, (list, tuple)):
      raise ValueError('Please pass a list or tuple to any function that'
                       ' expects a federated type placed at {};'
                       ' you passed {}'.format(type_spec.placement, value))
    else:
      return [
          to_representation_for_type(v, type_spec.member, callable_handler)
          for v in value
      ]
  else:
    raise NotImplementedError(
        'Unable to determine valid value representation for {} for what '
        'is currently an unsupported TFF type {}.'.format(
            str(value), str(type_spec)))


def stamp_computed_value_into_graph(value, graph):
  """Stamps `value` in `graph`.

  Args:
    value: An instance of `ComputedValue`.
    graph: The graph to stamp in.

  Returns:
    A Python object made of tensors stamped into `graph`, `tf.data.Dataset`s,
    and `anonymous_tuple.AnonymousTuple`s that structurally corresponds to the
    value passed at input.
  """
  if value is None:
    return None
  else:
    py_typecheck.check_type(value, ComputedValue)
    value = ComputedValue(
        to_representation_for_type(value.value, value.type_signature),
        value.type_signature)
    py_typecheck.check_type(graph, tf.Graph)
    if isinstance(value.type_signature, computation_types.TensorType):
      if isinstance(value.value, np.ndarray):
        value_type = computation_types.TensorType(
            tf.dtypes.as_dtype(value.value.dtype),
            tf.TensorShape(value.value.shape))
        type_utils.check_assignable_from(value.type_signature, value_type)
        with graph.as_default():
          return tf.constant(value.value)
      else:
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
    elif isinstance(value.type_signature, computation_types.SequenceType):
      return graph_utils.make_data_set_from_elements(
          graph, value.value, value.type_signature.element)
    else:
      raise NotImplementedError(
          'Unable to embed a computed value of type {} in graph.'.format(
              str(value.type_signature)))


def capture_computed_value_from_graph(value, type_spec):
  """Captures `value` from a TensorFlow graph.

  Args:
    value: A Python object made of tensors in `graph`, `tf.data.Dataset`s,
      `anonymous_tuple.AnonymousTuple`s and other structures, to be captured as
      an instance of `ComputedValue`.
    type_spec: The type of the value to be captured.

  Returns:
    An instance of `ComputedValue`.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)
  value = type_utils.to_canonical_value(value)
  return ComputedValue(to_representation_for_type(value, type_spec), type_spec)


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
    init_op, result = (
        tensorflow_deserialization.deserialize_and_call_tf_computation(
            comp.proto, stamped_arg, graph))
  with tf.Session(graph=graph) as sess:
    if init_op:
      sess.run(init_op)
    result_val = graph_utils.fetch_value_in_session(sess, result)
  return capture_computed_value_from_graph(result_val,
                                           comp.type_signature.result)


def numpy_cast(value, dtype, shape):
  """Returns a Numpy representation of `value` for given `dtype` and `shape`.

  Args:
    value: A tensor value (such as a numpy or a raw Python type).
    dtype: An instance of tf.DType.
    shape: An instance of tf.TensorShape.

  Returns:
    The Numpy represantation of `value` that matches `dtype` and `shape`.

  Raises:
    TypeError: If the `value` cannot be converted to the given `dtype` and the
      desired `shape`.
  """
  py_typecheck.check_type(dtype, tf.DType)
  py_typecheck.check_type(shape, tf.TensorShape)
  value_as_numpy_array = np.array(value, dtype=dtype.as_numpy_dtype)
  if list(value_as_numpy_array.shape) != shape.dims:
    raise TypeError('Expected shape {}, found {}.'.format(
        str(shape.dims), str(value_as_numpy_array.shape)))
  # NOTE: We don't want to make things more complicated than necessary by
  # returning the result as an array if it's just a plain scalar, so we
  # special-case this by pulling the singleton `np.ndarray`'s element out.
  if len(value_as_numpy_array.shape) > 0:  # pylint: disable=g-explicit-length-test
    return value_as_numpy_array
  else:
    return value_as_numpy_array.flatten()[0]


def multiply_by_scalar(value, multiplier):
  """Multiplies an instance of `ComputedValue` by a given scalar.

  Args:
    value: An instance of `ComputedValue` to multiply.
    multiplier: A scalar multipler.

  Returns:
    An instance of `ComputedValue` that represents the result of multiplication.
  """
  py_typecheck.check_type(value, ComputedValue)
  py_typecheck.check_type(multiplier, (float, np.float32))
  if isinstance(value.type_signature, computation_types.TensorType):
    result_val = numpy_cast(value.value * multiplier,
                            value.type_signature.dtype,
                            value.type_signature.shape)
    return ComputedValue(result_val, value.type_signature)
  elif isinstance(value.type_signature, computation_types.NamedTupleType):
    elements = anonymous_tuple.to_elements(value.value)
    type_elements = anonymous_tuple.to_elements(value.type_signature)
    result_elements = []
    for idx, (k, v) in enumerate(elements):
      multiplied_v = multiply_by_scalar(
          ComputedValue(v, type_elements[idx][1]), multiplier).value
      result_elements.append((k, multiplied_v))
    return ComputedValue(
        anonymous_tuple.AnonymousTuple(result_elements), value.type_signature)
  else:
    raise NotImplementedError(
        'Multiplying vlues of type {} by a scalar is unsupported.'.format(
            str(value.type_signature)))


def get_cardinalities(value):
  """Get a dictionary mapping placements to their cardinalities from `value`.

  Args:
    value: An instance of `ComputationValue`.

  Returns:
    A dictionary from placement literals to the cardinalities of each placement.
  """
  py_typecheck.check_type(value, ComputedValue)
  if isinstance(value.type_signature, computation_types.FederatedType):
    if value.type_signature.all_equal:
      return {}
    else:
      py_typecheck.check_type(value.value, list)
      return {value.type_signature.placement: len(value.value)}
  elif isinstance(
      value.type_signature,
      (computation_types.TensorType, computation_types.SequenceType,
       computation_types.AbstractType, computation_types.FunctionType,
       computation_types.PlacementType)):
    return {}
  elif isinstance(value.type_signature, computation_types.NamedTupleType):
    py_typecheck.check_type(value.value, anonymous_tuple.AnonymousTuple)
    result = {}
    for idx, (_, elem_type) in enumerate(
        anonymous_tuple.to_elements(value.type_signature)):
      for k, v in six.iteritems(
          get_cardinalities(ComputedValue(value.value[idx], elem_type))):
        if k not in result:
          result[k] = v
        elif result[k] != v:
          raise ValueError(
              'Mismatching cardinalities for {}: {} vs. {}.'.format(
                  str(k), str(result[k]), str(v)))
    return result
  else:
    raise NotImplementedError(
        'Unable to get cardinalities from a value of TFF type {}.'.format(
            str(value.type_signature)))


class ComputationContext(object):
  """Encapsulates context/state in which computations or parts thereof run."""

  def __init__(self,
               parent_context=None,
               local_symbols=None,
               cardinalities=None):
    """Constructs a new execution context.

    Args:
      parent_context: The parent context, or `None` if this is the root.
      local_symbols: The dictionary of local symbols defined in this context, or
        `None` if there are none. The keys (names) are of a string type, and the
        values (what the names bind to) are of type `ComputedValue`.
      cardinalities: Placements cardinalities, if defined.
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
    if cardinalities is not None:
      py_typecheck.check_type(cardinalities, dict)
      for k, v in six.iteritems(cardinalities):
        py_typecheck.check_type(k, placement_literals.PlacementLiteral)
        py_typecheck.check_type(v, int)
      self._cardinalities = cardinalities
    else:
      self._cardinalities = None

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

  def get_cardinality(self, placement):
    """Returns the cardinality for `placement`.

    Args:
      placement: The placement, for which to return cardinality.
    """
    py_typecheck.check_type(placement, placement_literals.PlacementLiteral)
    if self._cardinalities is not None and placement in self._cardinalities:
      return self._cardinalities[placement]
    elif self._parent_context is not None:
      return self._parent_context.get_cardinality(placement)
    else:
      raise ValueError('Unable to determine the cardinality for {}.'.format(
          str(placement)))


def fit_argument(arg, type_spec, context):
  """Fits the given argument `arg` to match the given parameter `type_spec`.

  Args:
    arg: The argument to fit, an instance of `ComputedValue`.
    type_spec: The type of the parameter to fit to, an instance of `tff.Type` or
      something convertible to it.
    context: The context in which to perform the fitting, either an instance of
      `ComputationContext`, or `None` if unspecified.

  Returns:
    An instance of `ComputationValue` with the payload from `arg`, but matching
    the `type_spec` in the given context.

  Raises:
    TypeError: If the types mismatch.
    ValueError: If the value is invalid or does not fit the requested type.
  """
  py_typecheck.check_type(arg, ComputedValue)
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)
  if context is not None:
    py_typecheck.check_type(context, ComputationContext)
  type_utils.check_assignable_from(type_spec, arg.type_signature)
  if arg.type_signature == type_spec:
    return arg
  elif isinstance(type_spec, computation_types.NamedTupleType):
    py_typecheck.check_type(arg.value, anonymous_tuple.AnonymousTuple)
    result_elements = []
    for idx, (elem_name, elem_type) in enumerate(
        anonymous_tuple.to_elements(type_spec)):
      elem_val = ComputedValue(arg.value[idx], arg.type_signature[idx])
      if elem_val != elem_type:
        elem_val = fit_argument(elem_val, elem_type, context)
      result_elements.append((elem_name, elem_val.value))
    return ComputedValue(
        anonymous_tuple.AnonymousTuple(result_elements), type_spec)
  elif isinstance(type_spec, computation_types.FederatedType):
    type_utils.check_federated_type(
        arg.type_signature, placement=type_spec.placement)
    if arg.type_signature.all_equal:
      member_val = ComputedValue(arg.value, arg.type_signature.member)
      if type_spec.member != arg.type_signature.member:
        member_val = fit_argument(member_val, type_spec.member, context)
      if type_spec.all_equal:
        return ComputedValue(member_val.value, type_spec)
      else:
        cardinality = context.get_cardinality(type_spec.placement)
        return ComputedValue([member_val.value for _ in range(cardinality)],
                             type_spec)
    elif type_spec.all_equal:
      raise TypeError('Cannot fit a non all-equal {} into all-equal {}.'.format(
          str(arg.type_signature), str(type_spec)))
    else:
      py_typecheck.check_type(arg.value, list)

      def _fit_member_val(x):
        x_val = ComputedValue(x, arg.type_signature.member)
        return fit_argument(x_val, type_spec.member, context).value

      return ComputedValue([_fit_member_val(x) for x in arg.value], type_spec)
  else:
    # TODO(b/113123634): Possibly add more conversions, e.g., for tensor types.
    return arg


class ReferenceExecutor(context_base.Context):
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

  def __init__(self, compiler=None):
    """Creates a reference executor.

    Args:
      compiler: The compiler pipeline to be used by this executor, or `None` if
        the executor is to run without one.
    """
    # TODO(b/113116813): Add a way to declare environmental bindings here,
    # e.g., a way to specify how data URIs are mapped to physical resources.

    if compiler is not None:
      py_typecheck.check_type(compiler, compiler_pipeline.CompilerPipeline)
    self._compiler = compiler
    self._intrinsic_method_dict = {
        intrinsic_defs.FEDERATED_AGGREGATE.uri:
            self._federated_aggregate,
        intrinsic_defs.FEDERATED_APPLY.uri:
            self._federated_apply,
        intrinsic_defs.FEDERATED_MEAN.uri:
            self._federated_mean,
        intrinsic_defs.FEDERATED_BROADCAST.uri:
            self._federated_broadcast,
        intrinsic_defs.FEDERATED_COLLECT.uri:
            self._federated_collect,
        intrinsic_defs.FEDERATED_MAP.uri:
            self._federated_map,
        intrinsic_defs.FEDERATED_REDUCE.uri:
            self._federated_reduce,
        intrinsic_defs.FEDERATED_SUM.uri:
            self._federated_sum,
        intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri:
            self._federated_value_at_clients,
        intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri:
            self._federated_value_at_server,
        intrinsic_defs.FEDERATED_WEIGHTED_MEAN.uri:
            self._federated_weighted_mean,
        intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri:
            self._federated_zip_at_clients,
        intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri:
            self._federated_zip_at_server,
        intrinsic_defs.GENERIC_PLUS.uri:
            self._generic_plus,
        intrinsic_defs.GENERIC_ZERO.uri:
            self._generic_zero,
        intrinsic_defs.SEQUENCE_MAP.uri:
            self._sequence_map,
        intrinsic_defs.SEQUENCE_REDUCE.uri:
            self._sequence_reduce,
        intrinsic_defs.SEQUENCE_SUM.uri:
            self._sequence_sum,
    }

  def ingest(self, arg, type_spec):

    def _handle_callable(func, func_type):
      py_typecheck.check_type(func, computation_base.Computation)
      type_utils.check_assignable_from(func.type_signature, func_type)
      return func

    return to_representation_for_type(arg, type_spec, _handle_callable)

  def invoke(self, func, arg):
    comp = self._compile(func)
    cardinalities = {}
    root_context = ComputationContext(cardinalities=cardinalities)
    computed_comp = self._compute(comp, root_context)
    type_utils.check_assignable_from(comp.type_signature,
                                     computed_comp.type_signature)
    if not isinstance(computed_comp.type_signature,
                      computation_types.FunctionType):
      if arg is not None:
        raise TypeError('Unexpected argument {}.'.format(str(arg)))
      else:
        return computed_comp.value
    else:
      if arg is not None:

        def _handle_callable(func, func_type):
          py_typecheck.check_type(func, computation_base.Computation)
          type_utils.check_assignable_from(func.type_signature, func_type)
          computed_func = self._compute(self._compile(func), root_context)
          return computed_func.value

        computed_arg = ComputedValue(
            to_representation_for_type(
                arg, computed_comp.type_signature.parameter, _handle_callable),
            computed_comp.type_signature.parameter)
        cardinalities.update(get_cardinalities(computed_arg))
      else:
        computed_arg = None
      result = computed_comp.value(computed_arg)
      py_typecheck.check_type(result, ComputedValue)
      type_utils.check_assignable_from(comp.type_signature.result,
                                       result.type_signature)
      return result.value

  def _compile(self, comp):
    """Compiles a `computation_base.Computation` to prepare it for execution.

    Args:
      comp: An instance of `computation_base.Computation`.

    Returns:
      An instance of `computation_building_blocks.ComputationBuildingBlock` that
      contains the compiled logic of `comp`.
    """
    py_typecheck.check_type(comp, computation_base.Computation)
    if self._compiler is not None:
      comp = self._compiler.compile(comp)
    return transformations.name_compiled_computations(
        computation_building_blocks.ComputationBuildingBlock.from_proto(
            computation_impl.ComputationImpl.get_proto(comp)))

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
      type_utils.check_assignable_from(computed_func.type_signature.parameter,
                                       computed_arg.type_signature)
      computed_arg = fit_argument(
          computed_arg, computed_func.type_signature.parameter, context)
    else:
      computed_arg = None
    result = computed_func.value(computed_arg)
    py_typecheck.check_type(result, ComputedValue)
    type_utils.check_assignable_from(computed_func.type_signature.result,
                                     result.type_signature)
    return result

  def _compute_tuple(self, comp, context):
    py_typecheck.check_type(comp, computation_building_blocks.Tuple)
    result_elements = []
    result_type_elements = []
    for k, v in anonymous_tuple.to_elements(comp):
      computed_v = self._compute(v, context)
      type_utils.check_assignable_from(v.type_signature,
                                       computed_v.type_signature)
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
    type_utils.check_assignable_from(comp.type_signature, result_type)
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
    my_method = self._intrinsic_method_dict.get(comp.uri)
    if my_method is not None:
      # The interpretation of `my_method` depends on whether the intrinsic
      # does or does not take arguments. If it does, the method accepts the
      # argument as a `ComputedValue` instance. Otherwise, if the intrinsic
      # is not a function, but a constant (such as `GENERIC_ZERO`), the
      # method accepts the type of the result.
      if isinstance(comp.type_signature, computation_types.FunctionType):
        arg_type = comp.type_signature.parameter
        return ComputedValue(
            lambda x: my_method(fit_argument(x, arg_type, context)),
            comp.type_signature)
      else:
        return my_method(comp.type_signature)
    else:
      raise NotImplementedError('Intrinsic {} is currently unsupported.'.format(
          comp.uri))

  def _compute_data(self, comp, context):
    py_typecheck.check_type(comp, computation_building_blocks.Data)
    raise NotImplementedError('Data is currently unsupported.')

  def _compute_placement(self, comp, context):
    py_typecheck.check_type(comp, computation_building_blocks.Placement)
    raise NotImplementedError('Placement is currently unsupported.')

  def _sequence_sum(self, arg):
    py_typecheck.check_type(arg.type_signature, computation_types.SequenceType)
    total = self._generic_zero(arg.type_signature.element)
    for v in arg.value:
      total = self._generic_plus(
          ComputedValue(
              anonymous_tuple.AnonymousTuple([(None, total.value), (None, v)]),
              [arg.type_signature.element, arg.type_signature.element]))
    return total

  def _federated_collect(self, arg):
    type_utils.check_federated_type(arg.type_signature, None,
                                    placements.CLIENTS, False)
    return ComputedValue(
        arg.value,
        computation_types.FederatedType(
            computation_types.SequenceType(arg.type_signature.member),
            placements.SERVER, True))

  def _federated_map(self, arg):
    mapping_type = arg.type_signature[0]
    py_typecheck.check_type(mapping_type, computation_types.FunctionType)
    type_utils.check_federated_type(arg.type_signature[1],
                                    mapping_type.parameter, placements.CLIENTS,
                                    False)
    fn = arg.value[0]
    result_val = [
        fn(ComputedValue(x, mapping_type.parameter)).value for x in arg.value[1]
    ]
    result_type = computation_types.FederatedType(mapping_type.result,
                                                  placements.CLIENTS, False)
    return ComputedValue(result_val, result_type)

  def _federated_apply(self, arg):
    mapping_type = arg.type_signature[0]
    py_typecheck.check_type(mapping_type, computation_types.FunctionType)
    type_utils.check_federated_type(
        arg.type_signature[1], mapping_type.parameter, placements.SERVER, True)
    fn = arg.value[0]
    result_val = fn(ComputedValue(arg.value[1], mapping_type.parameter)).value
    result_type = computation_types.FederatedType(mapping_type.result,
                                                  placements.SERVER, True)
    return ComputedValue(result_val, result_type)

  def _federated_sum(self, arg):
    type_utils.check_federated_type(arg.type_signature, None,
                                    placements.CLIENTS, False)
    collected_val = self._federated_collect(arg)
    federated_apply_arg = anonymous_tuple.AnonymousTuple(
        [(None, self._sequence_sum), (None, collected_val.value)])
    apply_fn_type = computation_types.FunctionType(
        computation_types.SequenceType(arg.type_signature.member),
        arg.type_signature.member)
    return self._federated_apply(
        ComputedValue(federated_apply_arg,
                      [apply_fn_type, collected_val.type_signature]))

  def _federated_value_at_clients(self, arg):
    return ComputedValue(
        arg.value,
        computation_types.FederatedType(
            arg.type_signature, placements.CLIENTS, all_equal=True))

  def _federated_value_at_server(self, arg):
    return ComputedValue(
        arg.value,
        computation_types.FederatedType(
            arg.type_signature, placements.SERVER, all_equal=True))

  def _generic_zero(self, type_spec):
    if isinstance(type_spec, computation_types.TensorType):
      # TODO(b/113116813): Replace this with something more efficient, probably
      # calling some helper method from Numpy.
      with tf.Graph().as_default() as graph:
        zeros = tf.constant(0, type_spec.dtype, type_spec.shape)
        with tf.Session(graph=graph) as sess:
          zeros_val = sess.run(zeros)
      return ComputedValue(zeros_val, type_spec)
    elif isinstance(type_spec, computation_types.NamedTupleType):
      return ComputedValue(
          anonymous_tuple.AnonymousTuple(
              [(k, self._generic_zero(v).value)
               for k, v in anonymous_tuple.to_elements(type_spec)]), type_spec)
    elif isinstance(
        type_spec,
        (computation_types.SequenceType, computation_types.FunctionType,
         computation_types.AbstractType, computation_types.PlacementType)):
      raise TypeError(
          'The generic_zero is not well-defined for TFF type {}.'.format(
              str(type_spec)))
    elif isinstance(type_spec, computation_types.FederatedType):
      if type_spec.all_equal:
        return ComputedValue(
            self._generic_zero(type_spec.member).value, type_spec)
      else:
        # TODO(b/113116813): Implement this in terms of the generic placement
        # operator once it's been added to the mix.
        raise NotImplementedError(
            'Generic zero support for non-all_equal federated types is not '
            'implemented yet.')
    else:
      raise NotImplementedError(
          'Generic zero support for {} is not implemented yet.'.format(
              str(type_spec)))

  def _generic_plus(self, arg):
    py_typecheck.check_type(arg.type_signature,
                            computation_types.NamedTupleType)
    if len(arg.type_signature) != 2:
      raise TypeError('Generic plus is undefined for tuples of size {}.'.format(
          str(len(arg.type_signature))))
    element_type = arg.type_signature[0]
    if arg.type_signature[1] != element_type:
      raise TypeError('Generic plus is undefined for two-tuples of different '
                      'types ({} vs. {}).'.format(
                          str(element_type), str(arg.type_signature[1])))
    if isinstance(element_type, computation_types.TensorType):
      return ComputedValue(arg.value[0] + arg.value[1], element_type)
    elif isinstance(element_type, computation_types.NamedTupleType):
      py_typecheck.check_type(arg.value[0], anonymous_tuple.AnonymousTuple)
      py_typecheck.check_type(arg.value[1], anonymous_tuple.AnonymousTuple)
      result_val_elements = []
      for idx, (name, elem_type) in enumerate(
          anonymous_tuple.to_elements(element_type)):
        to_add = ComputedValue(
            anonymous_tuple.AnonymousTuple([(None, arg.value[0][idx]),
                                            (None, arg.value[1][idx])]),
            [elem_type, elem_type])
        add_result = self._generic_plus(to_add)
        result_val_elements.append((name, add_result.value))
      return ComputedValue(
          anonymous_tuple.AnonymousTuple(result_val_elements), element_type)
    else:
      # TODO(b/113116813): Implement the remaining cases.
      raise NotImplementedError

  def _sequence_map(self, arg):
    mapping_type = arg.type_signature[0]
    py_typecheck.check_type(mapping_type, computation_types.FunctionType)
    sequence_type = arg.type_signature[1]
    py_typecheck.check_type(sequence_type, computation_types.SequenceType)
    type_utils.check_assignable_from(mapping_type.parameter,
                                     sequence_type.element)
    fn = arg.value[0]
    result_val = [
        fn(ComputedValue(x, mapping_type.parameter)).value for x in arg.value[1]
    ]
    result_type = computation_types.SequenceType(mapping_type.result)
    return ComputedValue(result_val, result_type)

  def _sequence_reduce(self, arg):
    py_typecheck.check_type(arg.type_signature,
                            computation_types.NamedTupleType)
    sequence_type = arg.type_signature[0]
    py_typecheck.check_type(sequence_type, computation_types.SequenceType)
    zero_type = arg.type_signature[1]
    op_type = arg.type_signature[2]
    py_typecheck.check_type(op_type, computation_types.FunctionType)
    type_utils.check_assignable_from(op_type.parameter,
                                     [zero_type, sequence_type.element])
    total = ComputedValue(arg.value[1], zero_type)
    reduce_fn = arg.value[2]
    for v in arg.value[0]:
      total = reduce_fn(
          ComputedValue(
              anonymous_tuple.AnonymousTuple([(None, total.value), (None, v)]),
              op_type.parameter))
    return total

  def _federated_reduce(self, arg):
    py_typecheck.check_type(arg.type_signature,
                            computation_types.NamedTupleType)
    federated_type = arg.type_signature[0]
    type_utils.check_federated_type(federated_type, None, placements.CLIENTS,
                                    False)
    zero_type = arg.type_signature[1]
    op_type = arg.type_signature[2]
    py_typecheck.check_type(op_type, computation_types.FunctionType)
    type_utils.check_assignable_from(op_type.parameter,
                                     [zero_type, federated_type.member])
    total = ComputedValue(arg.value[1], zero_type)
    reduce_fn = arg.value[2]
    for v in arg.value[0]:
      total = reduce_fn(
          ComputedValue(
              anonymous_tuple.AnonymousTuple([(None, total.value), (None, v)]),
              op_type.parameter))
    return self._federated_value_at_server(total)

  def _federated_mean(self, arg):
    type_utils.check_federated_type(arg.type_signature, None,
                                    placements.CLIENTS, False)
    py_typecheck.check_type(arg.value, list)
    server_sum = self._federated_sum(arg)
    unplaced_avg = multiply_by_scalar(
        ComputedValue(server_sum.value, server_sum.type_signature.member),
        1.0 / float(len(arg.value)))
    return ComputedValue(
        unplaced_avg.value,
        type_constructors.at_server(unplaced_avg.type_signature))

  def _federated_zip_at_server(self, arg):
    py_typecheck.check_type(arg.type_signature,
                            computation_types.NamedTupleType)
    for idx in range(len(arg.type_signature)):
      type_utils.check_federated_type(arg.type_signature[idx], None,
                                      placements.SERVER, True)
    return ComputedValue(
        arg.value,
        type_constructors.at_server(
            computation_types.NamedTupleType(
                [(k, v.member) if k else v.member
                 for k, v in anonymous_tuple.to_elements(arg.type_signature)])))

  def _federated_zip_at_clients(self, arg):
    py_typecheck.check_type(arg.type_signature,
                            computation_types.NamedTupleType)
    py_typecheck.check_type(arg.value, anonymous_tuple.AnonymousTuple)
    zip_args = []
    zip_arg_types = []
    for idx in range(len(arg.type_signature)):
      val = arg.value[idx]
      py_typecheck.check_type(val, list)
      zip_args.append(val)
      val_type = arg.type_signature[idx]
      type_utils.check_federated_type(val_type, None, placements.CLIENTS, False)
      zip_arg_types.append(val_type.member)
    zipped_val = [anonymous_tuple.from_container(x) for x in zip(*zip_args)]
    return ComputedValue(
        zipped_val,
        type_constructors.at_clients(
            computation_types.NamedTupleType(zip_arg_types)))

  def _federated_aggregate(self, arg):
    py_typecheck.check_type(arg.type_signature,
                            computation_types.NamedTupleType)
    if len(arg.type_signature) != 5:
      raise TypeError('Expected a 5-tuple, found {}.'.format(
          str(arg.type_signature)))
    root_accumulator = self._federated_reduce(
        ComputedValue(
            anonymous_tuple.from_container([arg.value[k] for k in range(3)]),
            [arg.type_signature[k] for k in range(3)]))
    return self._federated_apply(
        ComputedValue([arg.value[4], root_accumulator.value],
                      [arg.type_signature[4], root_accumulator.type_signature]))

  def _federated_weighted_mean(self, arg):
    py_typecheck.check_type(arg.type_signature,
                            computation_types.NamedTupleType)
    if len(arg.type_signature) != 2:
      raise TypeError('Expected a 2-tuple, found {}.'.format(
          str(arg.type_signature)))
    for _, v in anonymous_tuple.to_elements(arg.type_signature):
      type_utils.check_federated_type(v, None, placements.CLIENTS, False)
      if not type_utils.is_average_compatible(v.member):
        raise TypeError('Expected average-compatible args,'
                        ' got {} from argument of type {}.'.format(
                            str(v.member), arg.type_signature))
    v_type = arg.type_signature[0].member
    w_type = arg.type_signature[1].member
    py_typecheck.check_type(w_type, computation_types.TensorType)
    if w_type.shape.ndims != 0:
      raise TypeError('Expected scalar weight, got {}.'.format(str(w_type)))
    total = sum(arg.value[1])
    products_val = [
        multiply_by_scalar(ComputedValue(v, v_type), w / total).value
        for v, w in zip(arg.value[0], arg.value[1])
    ]
    return self._federated_sum(
        ComputedValue(products_val, type_constructors.at_clients(v_type)))

  def _federated_broadcast(self, arg):
    type_utils.check_federated_type(arg.type_signature, None, placements.SERVER,
                                    True)
    return ComputedValue(
        arg.value,
        computation_types.FederatedType(arg.type_signature.member,
                                        placements.CLIENTS, True))

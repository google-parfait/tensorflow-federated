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
"""A simple interpreted reference context.

This context is designed for simplicity, not for performance. It is intended
for use in unit tests, as the golden standard and point of comparison for other
contexts.
"""

import collections
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import compiler_pipeline
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import intrinsic_reductions
from tensorflow_federated.python.core.impl.compiler import transformations
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import cardinalities_utils
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


class ComputedValue(object):
  """A container for values computed by the reference context."""

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
    return 'ComputedValue({}, {})'.format(self._value, self._type_signature)


def to_canonical_value(value):
  """Converts a Python object to a canonical TFF value.

  Args:
    value: The object to convert.

  Returns:
    The canonical TFF representation of `value` for a given type.
  """
  if value is None:
    return None
  elif isinstance(value, dict):
    if isinstance(value, collections.OrderedDict):
      items = value.items()
    else:
      items = sorted(value.items())
    return structure.Struct((k, to_canonical_value(v)) for k, v in items)
  elif isinstance(value, (tuple, list)):
    return [to_canonical_value(e) for e in value]
  return value


def to_representation_for_type(value, type_spec, callable_handler=None):
  """Verifies or converts the `value` representation to match `type_spec`.

  This method first tries to determine whether `value` is a valid representation
  of TFF type `type_spec`. If so, it is returned unchanged. If not, but if it
  can be converted into a valid representation, it is converted to such, and the
  valid representation is returned. If no conversion to a valid representation
  is possible, TypeError is raised.

  The accepted forms of `value` for various TFF types are as follows:

  *   For TFF tensor types listed in
      `tensorflow_utils.TENSOR_REPRESENTATION_TYPES`.

  *   For TFF named tuple types, instances of `structure.Struct`.

  *   For TFF sequences, Python lists.

  *   For TFF functional types, Python callables that accept a single argument
      that is an instance of `ComputedValue` (if the function has a parameter)
      or `None` (otherwise), and return a `ComputedValue` instance as a result.
      This function only verifies that `value` is a callable.

  *   For TFF abstract types, there is no valid representation. The reference
      context requires all types in an executable computation to be concrete.

  *   For TFF placement types, the valid representations are the placement
      literals (currently only `tff.SERVER` and `tff.CLIENTS`).

  *   For TFF federated types with `all_equal` set to `True`, the representation
      is the same as the representation of the member constituent (thus, e.g.,
      a valid representation of `int32@SERVER` is the same as that of `int32`).
      For those types that have `all_equal_` set to `False`, the representation
      is a Python list of member constituents.

      Note: This function does not attempt at validating that the sizes of lists
      that represent federated values match the corresponding placemenets. The
      cardinality analysis is a separate step, handled by the reference context
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

  # Note: We do not simply call `type_conversions.infer_type()` on `value`, as
  # the representations of values in the reference context are only a subset of
  # the Python types recognized by that helper function.

  if type_spec.is_tensor():
    if tf.executing_eagerly() and isinstance(value, (tf.Tensor, tf.Variable)):
      value = value.numpy()
    py_typecheck.check_type(value, tensorflow_utils.TENSOR_REPRESENTATION_TYPES)
    inferred_type_spec = type_conversions.infer_type(value)
    if not type_spec.is_assignable_from(inferred_type_spec):
      raise TypeError(
          'The tensor type {} of the value representation does not match '
          'the type spec {}.'.format(inferred_type_spec, type_spec))
    return value
  elif type_spec.is_struct():
    type_spec_elements = structure.to_elements(type_spec)
    # Special-casing unodered dictionaries to allow their elements to be fed in
    # the order in which they're defined in the named tuple type.
    if (isinstance(value, dict) and
        (set(value.keys()) == set(k for k, _ in type_spec_elements))):
      value = collections.OrderedDict([
          (k, value[k]) for k, _ in type_spec_elements
      ])
    value = structure.from_container(value)
    value_elements = structure.to_elements(value)
    if len(value_elements) != len(type_spec_elements):
      raise TypeError(
          'The number of elements {} in the value tuple {} does not match the '
          'number of elements {} in the type spec {}.'.format(
              len(value_elements), value, len(type_spec_elements), type_spec))
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
    return structure.Struct(result_elements)
  elif type_spec.is_sequence():
    if isinstance(value, tf.data.Dataset):
      inferred_type_spec = computation_types.SequenceType(
          computation_types.to_type(value.element_spec))
      if not type_spec.is_assignable_from(inferred_type_spec):
        raise TypeError(
            'Value of type {!s} not assignable to expected type {!s}'.format(
                inferred_type_spec, type_spec))
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
  elif type_spec.is_function():
    if callable_handler is not None:
      return callable_handler(value, type_spec)
    else:
      raise TypeError(
          'Values that are callables have been explicitly disallowed '
          'in this context. If you would like to supply here a function '
          'as a parameter, please construct a computation that contains '
          'this call.')
  elif type_spec.is_abstract():
    raise TypeError(
        'Abstract types are not supported by the reference context.')
  elif type_spec.is_placement():
    py_typecheck.check_type(value, placements.PlacementLiteral)
    return value
  elif type_spec.is_federated():
    if type_spec.all_equal:
      return to_representation_for_type(value, type_spec.member,
                                        callable_handler)
    elif type_spec.placement is not placements.CLIENTS:
      raise TypeError(
          'Unable to determine a valid value representation for a federated '
          'type with non-equal members placed at {}.'.format(
              type_spec.placement))
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
        'is currently an unsupported TFF type {}.'.format(value, type_spec))


def stamp_computed_value_into_graph(
    value: Optional[ComputedValue],
    graph: tf.Graph,
) -> Any:
  """Stamps `value` in `graph`.

  Args:
    value: An optional `ComputedValue` to stamp into the graph.
    graph: The graph to stamp in.

  Returns:
    A Python object made of tensors stamped into `graph`, `tf.data.Dataset`s,
    and `structure.Struct`s that structurally corresponds to the
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
    if value.type_signature.is_tensor():
      if isinstance(value.value, np.ndarray):
        value_type = computation_types.TensorType(
            tf.dtypes.as_dtype(value.value.dtype),
            tf.TensorShape(value.value.shape))
        value.type_signature.check_assignable_from(value_type)
        with graph.as_default():
          return tf.constant(value.value)
      else:
        with graph.as_default():
          return tf.constant(
              value.value,
              dtype=value.type_signature.dtype,
              shape=value.type_signature.shape)
    elif value.type_signature.is_struct():
      elements = structure.to_elements(value.value)
      type_elements = structure.to_elements(value.type_signature)
      stamped_elements = []
      for idx, (k, v) in enumerate(elements):
        computed_v = ComputedValue(v, type_elements[idx][1])
        stamped_v = stamp_computed_value_into_graph(computed_v, graph)
        stamped_elements.append((k, stamped_v))
      return structure.Struct(stamped_elements)
    elif value.type_signature.is_sequence():
      return tensorflow_utils.make_data_set_from_elements(
          graph, value.value, value.type_signature.element)
    else:
      raise NotImplementedError(
          'Unable to embed a computed value of type {} in graph.'.format(
              value.type_signature))


def capture_computed_value_from_graph(value, type_spec):
  """Captures `value` from a TensorFlow graph.

  Args:
    value: A Python object made of tensors in `graph`, `tf.data.Dataset`s,
      `structure.Struct`s and other structures, to be captured as an instance of
      `ComputedValue`.
    type_spec: The type of the value to be captured.

  Returns:
    An instance of `ComputedValue`.
  """
  type_spec = computation_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, computation_types.Type)
  value = to_canonical_value(value)
  return ComputedValue(to_representation_for_type(value, type_spec), type_spec)


# TODO(b/139439722): Consolidate implementation to run a TF comp with an arg.
def run_tensorflow(comp, arg):
  """Runs a compiled TensorFlow computation `comp` with argument `arg`.

  Args:
    comp: An instance of `building_blocks.CompiledComputation` with embedded
      TensorFlow code.
    arg: An instance of `ComputedValue` that represents the argument, or `None`
      if the compuation expects no argument.

  Returns:
    An instance of `ComputedValue` with the result.
  """
  py_typecheck.check_type(comp, building_blocks.CompiledComputation)
  if arg is not None:
    py_typecheck.check_type(arg, ComputedValue)
  with tf.Graph().as_default() as graph:
    stamped_arg = stamp_computed_value_into_graph(arg, graph)
    init_op, result = (
        tensorflow_utils.deserialize_and_call_tf_computation(
            comp.proto, stamped_arg, graph))
  with tf.compat.v1.Session(graph=graph) as sess:
    if init_op:
      sess.run(init_op)
    result_val = tensorflow_utils.fetch_value_in_session(sess, result)
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
  if not (len(value_as_numpy_array.shape) == len(shape.dims) and
          all(value_as_numpy_array.shape[i] == shape.dims[i] or
              shape.dims[i].value is None) for i in range(len(shape.dims))):
    raise TypeError('Expected shape {}, found {}.'.format(
        shape.dims, value_as_numpy_array.shape))
  # Note: We don't want to make things more complicated than necessary by
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
  if value.type_signature.is_tensor():
    result_val = numpy_cast(value.value * multiplier,
                            value.type_signature.dtype,
                            value.type_signature.shape)
    return ComputedValue(result_val, value.type_signature)
  elif value.type_signature.is_struct():
    elements = structure.to_elements(value.value)
    type_elements = structure.to_elements(value.type_signature)
    result_elements = []
    for idx, (k, v) in enumerate(elements):
      multiplied_v = multiply_by_scalar(
          ComputedValue(v, type_elements[idx][1]), multiplier).value
      result_elements.append((k, multiplied_v))
    return ComputedValue(
        structure.Struct(result_elements), value.type_signature)
  else:
    raise NotImplementedError(
        'Multiplying vlues of type {} by a scalar is unsupported.'.format(
            value.type_signature))


class ComputationContext(object):
  """Encapsulates context/state in which computations or parts thereof run."""

  def __init__(self,
               parent_context: Optional['ComputationContext'] = None,
               local_symbols: Optional[Dict[str, ComputedValue]] = None,
               cardinalities: Optional[Dict[str, int]] = None):
    """Constructs a new execution context.

    Args:
      parent_context: The parent context, or `None` if this is the root.
      local_symbols: The dictionary of local symbols defined in this context, or
        `None` if there are none. The keys (names) are of a string type, and the
        values (what the names bind to) are of type `ComputedValue`.
      cardinalities: placements cardinalities, if defined.
    """
    if parent_context is not None:
      py_typecheck.check_type(parent_context, ComputationContext)
    self._parent_context = parent_context
    self._local_symbols = {}
    if local_symbols is not None:
      py_typecheck.check_type(local_symbols, dict)
      for k, v in local_symbols.items():
        py_typecheck.check_type(k, str)
        py_typecheck.check_type(v, ComputedValue)
        self._local_symbols[str(k)] = v
    if cardinalities is not None:
      py_typecheck.check_type(cardinalities, dict)
      for k, v in cardinalities.items():
        py_typecheck.check_type(k, placements.PlacementLiteral)
        py_typecheck.check_type(v, int)
      self._cardinalities = cardinalities
    else:
      self._cardinalities = None

  def resolve_reference(self, name: str) -> ComputedValue:
    """Resolves the given reference `name` in this context.

    Args:
      name: The string name to resolve.

    Returns:
      An instance of `ComputedValue` corresponding to this name.

    Raises:
      ValueError: If the name cannot be resolved.
    """
    py_typecheck.check_type(name, str)
    value = self._local_symbols.get(str(name))
    if value is not None:
      return value
    elif self._parent_context is not None:
      return self._parent_context.resolve_reference(name)
    else:
      raise ValueError(
          'The name \'{}\' is not defined in this context.'.format(name))

  def get_cardinality(self, placement: placements.PlacementLiteral) -> int:
    """Returns the cardinality for `placement`.

    Args:
      placement: The placement for which to return cardinality.
    """
    py_typecheck.check_type(placement, placements.PlacementLiteral)
    if self._cardinalities is not None and placement in self._cardinalities:
      return self._cardinalities[placement]
    elif self._parent_context is not None:
      return self._parent_context.get_cardinality(placement)
    else:
      raise ValueError(
          'Unable to determine the cardinality for {placement}. '
          'Consider adding an argument with placement {placement} to the '
          'top-level federated computation. This will allow the cardinality to '
          'be inferred automatically.'.format(placement=placement))


def fit_argument(arg: ComputedValue, type_spec,
                 context: Optional[ComputationContext]):
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
  type_spec.check_assignable_from(arg.type_signature)
  if arg.type_signature == type_spec:
    return arg
  elif type_spec.is_struct():
    py_typecheck.check_type(arg.value, structure.Struct)
    result_elements = []
    for idx, (elem_name,
              elem_type) in enumerate(structure.to_elements(type_spec)):
      elem_val = ComputedValue(arg.value[idx], arg.type_signature[idx])
      if elem_val != elem_type:
        elem_val = fit_argument(elem_val, elem_type, context)
      result_elements.append((elem_name, elem_val.value))
    return ComputedValue(structure.Struct(result_elements), type_spec)
  elif type_spec.is_federated():
    type_analysis.check_federated_type(
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
          arg.type_signature, type_spec))
    else:
      py_typecheck.check_type(arg.value, list)

      def _fit_member_val(x):
        x_val = ComputedValue(x, arg.type_signature.member)
        return fit_argument(x_val, type_spec.member, context).value

      return ComputedValue([_fit_member_val(x) for x in arg.value], type_spec)
  else:
    # TODO(b/113123634): Possibly add more conversions, e.g., for tensor types.
    return arg


class ReferenceContext(context_base.Context):
  """A simple interpreted reference context.

  This context is designed for simplicity and ease of reasoning about
  correctness, rather than for high performance. We will tolerate copying
  values, marshaling and marshaling when crossing TF graph boundary, etc., for
  the sake of keeping the logic minimal. The context can be reused across
  multiple calls, so any state associated with individual executions is
  maintained separately from this class. High-performance simulations on large
  data sets will require a separate context optimized for performance. This
  context is plugged in as the handler of computation invocations at the top
  level of the context stack.

  Note: The `tff.federated_secure_sum()` intrinsic is implemented using a
  non-secure algorithm in order to enable testing of the semantics of federated
  computaitons using the  secure sum intrinsic.
  """

  def __init__(self):
    """Creates a reference context."""

    # TODO(b/113116813): Add a way to declare environmental bindings here,
    # e.g., a way to specify how data URIs are mapped to physical resources.

    def _compilation_fn(
        comp: computation_base.Computation) -> computation_base.Computation:
      proto = computation_impl.ComputationImpl.get_proto(comp)
      bb_to_transform = building_blocks.ComputationBuildingBlock.from_proto(
          proto)
      intrinsics_reduced, _ = intrinsic_reductions.replace_intrinsics_with_bodies(
          bb_to_transform)
      dupes_removed, _ = transformations.remove_duplicate_building_blocks(
          intrinsics_reduced)
      comp_to_return = computation_impl.ComputationImpl(
          dupes_removed.proto, context_stack_impl.context_stack)
      return comp_to_return

    self._compiler = compiler_pipeline.CompilerPipeline(_compilation_fn)
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
        intrinsic_defs.FEDERATED_EVAL_AT_CLIENTS.uri:
            self._federated_eval_at_clients,
        intrinsic_defs.FEDERATED_EVAL_AT_SERVER.uri:
            self._federated_eval_at_server,
        intrinsic_defs.FEDERATED_MAP.uri:
            self._federated_map,
        intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri:
            self._federated_map_all_equal,
        intrinsic_defs.FEDERATED_SECURE_SUM.uri:
            self._federated_secure_sum,
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

    def _handle_callable(fn, fn_type):
      py_typecheck.check_type(fn, computation_base.Computation)
      fn.type_signature.check_assignable_from(fn_type)
      return fn

    return to_representation_for_type(arg, type_spec, _handle_callable)

  def invoke(self, fn, arg):
    comp = self._compile(fn)
    cardinalities = {placements.SERVER: 1}
    root_context = ComputationContext(cardinalities=cardinalities)
    if arg is not None:

      def _handle_callable(fn, fn_type):
        py_typecheck.check_type(fn, computation_base.Computation)
        fn.type_signature.check_assignable_from(fn_type)
        computed_fn = self._compute(self._compile(fn), root_context)
        return computed_fn.value

      computed_arg = ComputedValue(
          to_representation_for_type(arg, comp.type_signature.parameter,
                                     _handle_callable),
          comp.type_signature.parameter)
      cardinalities.update(
          cardinalities_utils.infer_cardinalities(computed_arg.value,
                                                  computed_arg.type_signature))
    else:
      computed_arg = None
    computed_comp = self._compute(comp, root_context)
    comp.type_signature.check_assignable_from(computed_comp.type_signature)

    if not computed_comp.type_signature.is_function():
      if computed_arg is not None:
        raise TypeError('Unexpected argument {}.'.format(arg))
      else:
        value = computed_comp.value
        result_type = fn.type_signature.result
        return type_conversions.type_to_py_container(value, result_type)
    else:
      result = computed_comp.value(computed_arg)
      py_typecheck.check_type(result, ComputedValue)
      comp.type_signature.result.check_assignable_from(result.type_signature)
      value = result.value
      fn_result_type = fn.type_signature.result
      return type_conversions.type_to_py_container(value, fn_result_type)

  def _compile(self, comp):
    """Compiles a `computation_base.Computation` to prepare it for execution.

    Args:
      comp: An instance of `computation_base.Computation`.

    Returns:
      An instance of `building_blocks.ComputationBuildingBlock` that
      contains the compiled logic of `comp`.
    """
    py_typecheck.check_type(comp, computation_base.Computation)
    if self._compiler is not None:
      comp = self._compiler.compile(comp)
    comp, _ = tree_transformations.uniquify_compiled_computation_names(
        building_blocks.ComputationBuildingBlock.from_proto(
            computation_impl.ComputationImpl.get_proto(comp)))
    return comp

  def _compute(self, comp, context):
    """Computes `comp` and returns the resulting computed value.

    Args:
      comp: An instance of `building_blocks.ComputationBuildingBlock`.
      context: An instance of `ComputationContext`.

    Returns:
      The corresponding instance of `ComputedValue` that represents the result
      of `comp`.

    Raises:
      TypeError: If type mismatch occurs during the course of computation.
      ValueError: If a malformed value is encountered.
      NotImplementedError: For computation building blocks that are not yet
        supported by this context.
    """
    if comp.is_compiled_computation():
      return self._compute_compiled(comp, context)
    elif comp.is_call():
      return self._compute_call(comp, context)
    elif comp.is_struct():
      return self._compute_tuple(comp, context)
    elif comp.is_reference():
      return self._compute_reference(comp, context)
    elif comp.is_selection():
      return self._compute_selection(comp, context)
    elif comp.is_lambda():
      return self._compute_lambda(comp, context)
    elif comp.is_block():
      return self._compute_block(comp, context)
    elif comp.is_intrinsic():
      return self._compute_intrinsic(comp, context)
    elif comp.is_data():
      return self._compute_data(comp, context)
    elif comp.is_placement():
      return self._compute_placement(comp, context)
    else:
      raise NotImplementedError(
          'A computation building block of a type {} not currently recognized '
          'by the reference context: {}.'.format(type(comp), comp))

  def _compute_compiled(self, comp, context):
    py_typecheck.check_type(comp, building_blocks.CompiledComputation)
    computation_oneof = comp.proto.WhichOneof('computation')
    if computation_oneof != 'tensorflow':
      raise ValueError(
          'Expected all parsed compiled computations to be tensorflow, '
          'but found \'{}\' instead.'.format(computation_oneof))
    else:
      return ComputedValue(lambda x: run_tensorflow(comp, x),
                           comp.type_signature)

  def _compute_call(self, comp, context):
    py_typecheck.check_type(comp, building_blocks.Call)
    computed_fn = self._compute(comp.function, context)
    py_typecheck.check_type(computed_fn.type_signature,
                            computation_types.FunctionType)
    if comp.argument is not None:
      computed_arg = self._compute(comp.argument, context)
      computed_fn.type_signature.parameter.check_assignable_from(
          computed_arg.type_signature)
      computed_arg = fit_argument(computed_arg,
                                  computed_fn.type_signature.parameter, context)
    else:
      computed_arg = None
    result = computed_fn.value(computed_arg)
    py_typecheck.check_type(result, ComputedValue)
    computed_fn.type_signature.result.check_assignable_from(
        result.type_signature)
    return result

  def _compute_tuple(self, comp, context):
    py_typecheck.check_type(comp, building_blocks.Struct)
    result_elements = []
    result_type_elements = []
    for k, v in structure.iter_elements(comp):
      computed_v = self._compute(v, context)
      v.type_signature.check_assignable_from(computed_v.type_signature)
      result_elements.append((k, computed_v.value))
      result_type_elements.append((k, computed_v.type_signature))
    return ComputedValue(
        structure.Struct(result_elements),
        computation_types.StructType([
            (k, v) if k else v for k, v in result_type_elements
        ]))

  def _compute_selection(self, comp, context):
    py_typecheck.check_type(comp, building_blocks.Selection)
    source = self._compute(comp.source, context)
    py_typecheck.check_type(source.type_signature, computation_types.StructType)
    py_typecheck.check_type(source.value, structure.Struct)
    if comp.name is not None:
      result_value = getattr(source.value, comp.name)
      result_type = getattr(source.type_signature, comp.name)
    else:
      assert comp.index is not None
      result_value = source.value[comp.index]
      result_type = source.type_signature[comp.index]
    comp.type_signature.check_assignable_from(result_type)
    return ComputedValue(result_value, result_type)

  def _compute_lambda(self, comp, context):
    py_typecheck.check_type(comp, building_blocks.Lambda)
    py_typecheck.check_type(context, ComputationContext)

    def _wrap(arg):
      """Wraps `context` in a new context which sets the parameter's value."""
      if comp.parameter_type is None:
        if arg is not None:
          raise TypeError(
              'No-argument lambda called with argument of type {}.'.format(
                  arg.type_signature))
        return context
      py_typecheck.check_type(arg, ComputedValue)
      if not comp.parameter_type.is_assignable_from(arg.type_signature):
        raise TypeError(
            'Expected the type of argument {} to be {}, found {}.'.format(
                comp.parameter_name, comp.parameter_type, arg.type_signature))
      return ComputationContext(context, {comp.parameter_name: arg})

    return ComputedValue(lambda x: self._compute(comp.result, _wrap(x)),
                         comp.type_signature)

  def _compute_reference(self, comp, context):
    py_typecheck.check_type(comp, building_blocks.Reference)
    py_typecheck.check_type(context, ComputationContext)
    return context.resolve_reference(comp.name)

  def _compute_block(self, comp, context):
    py_typecheck.check_type(comp, building_blocks.Block)
    py_typecheck.check_type(context, ComputationContext)
    for local_name, local_comp in comp.locals:
      local_val = self._compute(local_comp, context)
      context = ComputationContext(context, {local_name: local_val})
    return self._compute(comp.result, context)

  def _compute_intrinsic(self, comp, context):
    py_typecheck.check_type(comp, building_blocks.Intrinsic)
    my_method = self._intrinsic_method_dict.get(comp.uri)
    if my_method is not None:
      # The interpretation of `my_method` depends on whether the intrinsic
      # does or does not take arguments. If it does, the method accepts the
      # argument as a `ComputedValue` instance. Otherwise, if the intrinsic
      # is not a function, but a constant (such as `GENERIC_ZERO`), the
      # method accepts the type of the result.
      if comp.type_signature.is_function():
        arg_type = comp.type_signature.parameter
        return ComputedValue(
            lambda x: my_method(fit_argument(x, arg_type, context), context),
            comp.type_signature)
      else:
        return my_method(comp.type_signature, context)
    else:
      raise NotImplementedError('Intrinsic {} is currently unsupported.'.format(
          comp.uri))

  def _compute_data(self, comp, context):
    py_typecheck.check_type(comp, building_blocks.Data)
    raise NotImplementedError('Data is currently unsupported.')

  def _compute_placement(self, comp, context):
    py_typecheck.check_type(comp, building_blocks.Placement)
    raise NotImplementedError('Placement is currently unsupported.')

  def _sequence_sum(self, arg, context):
    del context  # Unused (left as arg b.c. functions must have same shape).
    inferred_type_spec = type_conversions.infer_type(arg.value[0])
    py_typecheck.check_type(arg.type_signature, computation_types.SequenceType)
    total = self._generic_zero(inferred_type_spec)
    for v in arg.value:
      total = self._generic_plus(
          ComputedValue(
              structure.Struct([(None, total.value), (None, v)]),
              [arg.type_signature.element, arg.type_signature.element]))
    return total

  def _federated_collect(self, arg, context):
    del context  # Unused (left as arg b.c. functions must have same shape).
    type_analysis.check_federated_type(arg.type_signature, None,
                                       placements.CLIENTS, False)
    return ComputedValue(
        arg.value,
        computation_types.FederatedType(
            computation_types.SequenceType(arg.type_signature.member),
            placements.SERVER, True))

  def _federated_eval_shared(
      self,
      arg: ComputedValue,
      context: ComputationContext,
      placement: placements.PlacementLiteral,
      all_equal: bool,
  ) -> ComputedValue:
    py_typecheck.check_type(arg, ComputedValue)
    py_typecheck.check_type(arg.type_signature, computation_types.FunctionType)
    if arg.type_signature.parameter is not None:
      raise TypeError(
          'Expected federated_eval parameter to be `None`, found {}.'.format(
              arg.type_signature.parameter))
    cardinality = context.get_cardinality(placement)
    fn_to_eval = arg.value
    if cardinality == 1:
      value = fn_to_eval(None).value
    else:
      value = [fn_to_eval(None).value for _ in range(cardinality)]
    return ComputedValue(
        value,
        computation_types.FederatedType(
            arg.type_signature.result, placement, all_equal=all_equal))

  def _federated_eval_at_clients(self, arg, context):
    return self._federated_eval_shared(arg, context, placements.CLIENTS, False)

  def _federated_eval_at_server(self, arg, context):
    return self._federated_eval_shared(arg, context, placements.SERVER, True)

  def _federated_map(self, arg, context):
    del context  # Unused (left as arg b.c. functions must have same shape).
    mapping_type = arg.type_signature[0]
    py_typecheck.check_type(mapping_type, computation_types.FunctionType)
    type_analysis.check_federated_type(arg.type_signature[1],
                                       mapping_type.parameter,
                                       placements.CLIENTS, False)
    fn = arg.value[0]
    result_val = [
        fn(ComputedValue(x, mapping_type.parameter)).value for x in arg.value[1]
    ]
    result_type = computation_types.FederatedType(mapping_type.result,
                                                  placements.CLIENTS, False)
    return ComputedValue(result_val, result_type)

  def _federated_map_all_equal(self, arg, context):
    del context  # Unused (left as arg b.c. functions must have same shape)
    mapping_type = arg.type_signature[0]
    py_typecheck.check_type(mapping_type, computation_types.FunctionType)
    type_analysis.check_federated_type(
        arg.type_signature[1],
        mapping_type.parameter,
        placements.CLIENTS,
        all_equal=True)
    fn = arg.value[0]
    result_val = fn(ComputedValue(arg.value[1], mapping_type.parameter)).value
    result_type = computation_types.FederatedType(
        mapping_type.result, placements.CLIENTS, all_equal=True)
    return ComputedValue(result_val, result_type)

  def _federated_apply(self, arg, context):
    del context  # Unused (left as arg b.c. functions must have same shape)
    mapping_type = arg.type_signature[0]
    py_typecheck.check_type(mapping_type, computation_types.FunctionType)
    type_analysis.check_federated_type(arg.type_signature[1],
                                       mapping_type.parameter,
                                       placements.SERVER, True)
    fn = arg.value[0]
    result_val = fn(ComputedValue(arg.value[1], mapping_type.parameter)).value
    result_type = computation_types.FederatedType(mapping_type.result,
                                                  placements.SERVER, True)
    return ComputedValue(result_val, result_type)

  def _federated_secure_sum(self, arg, context):
    py_typecheck.check_type(arg.type_signature, computation_types.StructType)
    py_typecheck.check_len(arg.type_signature, 2)
    value = ComputedValue(arg.value[0], arg.type_signature[0])
    return self._federated_sum(value, context)

  def _federated_sum(self, arg, context):
    type_analysis.check_federated_type(arg.type_signature, None,
                                       placements.CLIENTS, False)
    collected_val = self._federated_collect(arg, context)
    federated_apply_arg = structure.from_container(
        (lambda arg: self._sequence_sum(arg, context), collected_val.value))
    apply_fn_type = computation_types.FunctionType(
        computation_types.SequenceType(arg.type_signature.member),
        arg.type_signature.member)
    return self._federated_apply(
        ComputedValue(federated_apply_arg,
                      [apply_fn_type, collected_val.type_signature]), context)

  def _federated_value_at_clients(self, arg, context):
    del context  # Unused (left as arg b.c. functions must have same shape)
    return ComputedValue(
        arg.value,
        computation_types.FederatedType(
            arg.type_signature, placements.CLIENTS, all_equal=True))

  def _federated_value_at_server(self, arg, context):
    del context  # Unused (left as arg b.c. functions must have same shape)
    return ComputedValue(
        arg.value,
        computation_types.FederatedType(
            arg.type_signature, placements.SERVER, all_equal=True))

  def _generic_zero(self, type_spec):
    if type_spec.is_tensor():
      # TODO(b/113116813): Replace this with something more efficient, probably
      # calling some helper method from Numpy.
      with tf.Graph().as_default() as graph:
        zeros = tf.constant(0, type_spec.dtype, type_spec.shape)
        with tf.compat.v1.Session(graph=graph) as sess:
          zeros_val = sess.run(zeros)
      return ComputedValue(zeros_val, type_spec)
    elif type_spec.is_struct():
      type_elements_iter = structure.iter_elements(type_spec)
      return ComputedValue(
          structure.Struct(
              (k, self._generic_zero(v).value) for k, v in type_elements_iter),
          type_spec)
    elif (type_spec.is_sequence() or type_spec.is_function() or
          type_spec.is_abstract() or type_spec.is_placement()):
      raise TypeError(
          'The generic_zero is not well-defined for TFF type {}.'.format(
              type_spec))
    elif type_spec.is_federated():
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
              type_spec))

  def _generic_plus(self, arg):
    py_typecheck.check_type(arg.type_signature, computation_types.StructType)
    if len(arg.type_signature) != 2:
      raise TypeError('Generic plus is undefined for tuples of size {}.'.format(
          len(arg.type_signature)))
    element_type = arg.type_signature[0]
    if arg.type_signature[1] != element_type:
      raise TypeError('Generic plus is undefined for two-tuples of different '
                      'types ({} vs. {}).'.format(element_type,
                                                  arg.type_signature[1]))
    if element_type.is_tensor():
      val = numpy_cast(arg.value[0] + arg.value[1], element_type.dtype,
                       element_type.shape)
      return ComputedValue(val, element_type)
    elif element_type.is_struct():
      py_typecheck.check_type(arg.value[0], structure.Struct)
      py_typecheck.check_type(arg.value[1], structure.Struct)
      result_val_elements = []
      for idx, (name,
                elem_type) in enumerate(structure.to_elements(element_type)):
        to_add = ComputedValue(
            structure.Struct([(None, arg.value[0][idx]),
                              (None, arg.value[1][idx])]),
            [elem_type, elem_type])
        add_result = self._generic_plus(to_add)
        result_val_elements.append((name, add_result.value))
      return ComputedValue(structure.Struct(result_val_elements), element_type)
    else:
      # TODO(b/113116813): Implement the remaining cases, e.g. federated
      # types like int32@SERVER.
      raise NotImplementedError(
          'Generic plus not supported for elements of type {}, e.g. {}.'
          'Please file an issue on GitHub if you need this type supported'
          .format(element_type, arg.value[0]))

  def _sequence_map(self, arg, context):
    del context  # Unused (left as arg b.c. functions must have same shape)
    mapping_type = arg.type_signature[0]
    py_typecheck.check_type(mapping_type, computation_types.FunctionType)
    sequence_type = arg.type_signature[1]
    py_typecheck.check_type(sequence_type, computation_types.SequenceType)
    mapping_type.parameter.check_assignable_from(sequence_type.element)
    fn = arg.value[0]
    result_val = [
        fn(ComputedValue(x, mapping_type.parameter)).value for x in arg.value[1]
    ]
    result_type = computation_types.SequenceType(mapping_type.result)
    return ComputedValue(result_val, result_type)

  def _sequence_reduce(self, arg, context):
    del context  # Unused (left as arg b.c. functions must have same shape)
    py_typecheck.check_type(arg.type_signature, computation_types.StructType)
    sequence_type = arg.type_signature[0]
    py_typecheck.check_type(sequence_type, computation_types.SequenceType)
    zero_type = arg.type_signature[1]
    op_type = arg.type_signature[2]
    py_typecheck.check_type(op_type, computation_types.FunctionType)
    op_type.parameter.check_assignable_from(
        computation_types.StructType([zero_type, sequence_type.element]))
    total = ComputedValue(arg.value[1], zero_type)
    reduce_fn = arg.value[2]
    for v in arg.value[0]:
      total = reduce_fn(
          ComputedValue(
              structure.Struct([(None, total.value), (None, v)]),
              op_type.parameter))
    return total

  def _federated_mean(self, arg, context):
    type_analysis.check_federated_type(arg.type_signature, None,
                                       placements.CLIENTS, False)
    py_typecheck.check_type(arg.value, list)
    server_sum = self._federated_sum(arg, context)
    unplaced_avg = multiply_by_scalar(
        ComputedValue(server_sum.value, server_sum.type_signature.member),
        1.0 / float(len(arg.value)))
    return ComputedValue(
        unplaced_avg.value,
        computation_types.at_server(unplaced_avg.type_signature))

  def _federated_zip_at_server(self, arg, context):
    del context  # Unused (left as arg b.c. functions must have same shape)
    py_typecheck.check_type(arg.type_signature, computation_types.StructType)
    for idx in range(len(arg.type_signature)):
      type_analysis.check_federated_type(arg.type_signature[idx], None,
                                         placements.SERVER, True)
    return ComputedValue(
        arg.value,
        computation_types.at_server(
            computation_types.StructType([
                (k, v.member) if k else v.member
                for k, v in structure.iter_elements(arg.type_signature)
            ])))

  def _federated_zip_at_clients(self, arg, context):
    del context  # Unused (left as arg b.c. functions must have same shape)
    py_typecheck.check_type(arg.type_signature, computation_types.StructType)
    py_typecheck.check_type(arg.value, structure.Struct)
    zip_args = []
    zip_arg_types = []
    for idx in range(len(arg.type_signature)):
      val = arg.value[idx]
      py_typecheck.check_type(val, list)
      zip_args.append(val)
      val_type = arg.type_signature[idx]
      type_analysis.check_federated_type(val_type, None, placements.CLIENTS,
                                         False)
      zip_arg_types.append(val_type.member)
    zipped_val = [structure.from_container(x) for x in zip(*zip_args)]
    return ComputedValue(
        zipped_val,
        computation_types.at_clients(
            computation_types.StructType(zip_arg_types)))

  def _federated_aggregate(self, arg, context):
    py_typecheck.check_type(arg.type_signature, computation_types.StructType)
    if len(arg.type_signature) != 5:
      raise TypeError('Expected a 5-tuple, found {}.'.format(
          arg.type_signature))
    values, zero, reduce, merge, report = arg.value
    values_type, zero_type, reduce_type, merge_type, report_type = arg.type_signature
    del merge, merge_type
    type_analysis.check_federated_type(values_type, None, placements.CLIENTS,
                                       False)
    reduce_type.check_function()
    reduce_type.parameter.check_assignable_from(
        computation_types.StructType([zero_type, values_type.member]))
    total = ComputedValue(zero, zero_type)
    for v in values:
      total = reduce(
          ComputedValue(
              structure.Struct([(None, total.value), (None, v)]),
              reduce_type.parameter))
    root_accumulator = self._federated_value_at_server(total, context)

    return self._federated_apply(
        ComputedValue([report, root_accumulator.value],
                      [report_type, root_accumulator.type_signature]), context)

  def _federated_weighted_mean(self, arg, context):
    type_analysis.check_valid_federated_weighted_mean_argument_tuple_type(
        arg.type_signature)
    v_type = arg.type_signature[0].member
    total = sum(arg.value[1])
    products_val = [
        multiply_by_scalar(ComputedValue(v, v_type), w / total).value
        for v, w in zip(arg.value[0], arg.value[1])
    ]
    return self._federated_sum(
        ComputedValue(products_val, computation_types.at_clients(v_type)),
        context)

  def _federated_broadcast(self, arg, context):
    del context  # Unused (left as arg b.c. functions must have same shape)
    type_analysis.check_federated_type(arg.type_signature, None,
                                       placements.SERVER, True)
    return ComputedValue(
        arg.value,
        computation_types.FederatedType(arg.type_signature.member,
                                        placements.CLIENTS, True))


def create_reference_context():
  return ReferenceContext()


def set_reference_context():
  """Sets a reference context that executes computations locally."""
  context = ReferenceContext()
  context_stack_impl.context_stack.set_default_context(context)

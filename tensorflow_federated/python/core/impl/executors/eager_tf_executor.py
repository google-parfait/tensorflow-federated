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
"""A simple executor that operates synchronously in eager TensorFlow mode."""

from typing import Any, MutableMapping, Optional

import cachetools
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import typed_object
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils
from tensorflow_federated.python.tensorflow_libs import graph_merge


# Cache size here is simply heuristic, no formal analysis.
_TF_FUNCTION_CACHE_SIZE = 100


def _get_wrapped_function_from_comp(comp, must_pin_function_to_cpu, param_type,
                                    device):
  """Extracts the TensorFlow function from serialized computation.

  Args:
    comp: An instance of `pb.Computation`.
    must_pin_function_to_cpu: A boolean flag to indicate if the computation is
      forced to be on CPUs.
    param_type: A `tff.Type` instance or None.
    device: A `tf.config.LogicalDevice` or None.

  Returns:
    A TensorFlow ConcreteFunction.
  """

  def function_to_wrap():
    """No-arg function to import graph def.

    We pass a no-arg function to `tf.compat.v1.wrap_function` to avoid
    the leftover placeholders that can result from binding arguments to the
    imported graphdef via `input_map`. The correct signature will be added to
    this function later, via the `prune` call below.

    Returns:
      Result of importing graphdef backing `comp`.
    """
    graph_def = serialization_utils.unpack_graph_def(comp.tensorflow.graph_def)
    init_op = comp.tensorflow.initialize_op
    if init_op:
      graph_def = tensorflow_utils.add_control_deps_for_init_op(
          graph_def, init_op)

    def _import_fn():
      return tf.import_graph_def(
          graph_merge.uniquify_shared_names(graph_def), name='')

    if must_pin_function_to_cpu:
      with tf.device('cpu'):
        return _import_fn()
    elif device is not None:
      with tf.device(device.name):
        return _import_fn()
    else:
      return _import_fn()

  wrapped_noarg_fn = tf.compat.v1.wrap_function(function_to_wrap, signature=[])

  if param_type is not None:
    input_tensor_names = tensorflow_utils.extract_tensor_names_from_binding(
        comp.tensorflow.parameter)
  else:
    input_tensor_names = []
  output_tensor_names = tensorflow_utils.extract_tensor_names_from_binding(
      comp.tensorflow.result)
  import_graph = wrapped_noarg_fn.graph
  try:
    wrapped_fn = wrapped_noarg_fn.prune(
        feeds=tf.nest.map_structure(import_graph.as_graph_element,
                                    input_tensor_names),
        fetches=tf.nest.map_structure(import_graph.as_graph_element,
                                      output_tensor_names),
    )
  except KeyError as e:
    raise TypeError(
        'Caught exception trying to prune graph `{g}` with '
        'feeds {feeds} and fetches {fetches}. This indicates that these '
        'names may not refer to tensors in the graph. .\nException: {e}'.format(
            g=import_graph,
            feeds=input_tensor_names,
            fetches=output_tensor_names,
            e=e))
  return wrapped_fn


def embed_tensorflow_computation(comp, type_spec=None, device=None):
  """Embeds a TensorFlow computation for use in the eager context.

  Args:
    comp: An instance of `pb.Computation`.
    type_spec: An optional `tff.Type` instance or something convertible to it.
    device: An optional `tf.config.LogicalDevice`.

  Returns:
    Either a one-argument or a zero-argument callable that executes the
    computation in eager mode.

  Raises:
    TypeError: If arguments are of the wrong types, e.g., in `comp` is not a
      TensorFlow computation.
  """
  # TODO(b/134543154): Decide whether this belongs in `tensorflow_utils.py`
  # since it deals exclusively with eager mode. Incubate here, and potentially
  # move there, once stable.

  py_typecheck.check_type(comp, pb.Computation)
  comp_type = type_serialization.deserialize_type(comp.type)
  type_spec = computation_types.to_type(type_spec)
  if type_spec is not None:
    if not type_spec.is_equivalent_to(comp_type):
      raise TypeError('Expected a computation of type {}, got {}.'.format(
          type_spec, comp_type))
  else:
    type_spec = comp_type
  # TODO(b/155198591): Currently, TF will raise on any function returning a
  # `tf.data.Dataset` not pinned to CPU. We should follow up here and remove
  # this gating when we can.
  must_pin_function_to_cpu = type_analysis.contains(type_spec.result,
                                                    lambda t: t.is_sequence())
  which_computation = comp.WhichOneof('computation')
  if which_computation != 'tensorflow':
    raise TypeError('Expected a TensorFlow computation, found {}.'.format(
        which_computation))

  if type_spec.is_function():
    param_type = type_spec.parameter
    result_type = type_spec.result
  else:
    param_type = None
    result_type = type_spec

  wrapped_fn = _get_wrapped_function_from_comp(comp, must_pin_function_to_cpu,
                                               param_type, device)

  param_fns = []
  if param_type is not None:
    for spec in anonymous_tuple.flatten(type_spec.parameter):
      if spec.is_tensor():
        param_fns.append(lambda x: x)
      else:
        py_typecheck.check_type(spec, computation_types.SequenceType)
        param_fns.append(tf.data.experimental.to_variant)

  result_fns = []
  for spec in anonymous_tuple.flatten(result_type):
    if spec.is_tensor():
      result_fns.append(lambda x: x)
    else:
      py_typecheck.check_type(spec, computation_types.SequenceType)
      structure = type_conversions.type_to_tf_structure(spec.element)

      def fn(x, structure=structure):
        return tf.data.experimental.from_variant(x, structure)

      result_fns.append(fn)

  def _fn_to_return(arg, param_fns, wrapped_fn):  # pylint:disable=missing-docstring
    param_elements = []
    if arg is not None:
      arg_parts = anonymous_tuple.flatten(arg)
      if len(arg_parts) != len(param_fns):
        raise RuntimeError('Expected {} arguments, found {}.'.format(
            len(param_fns), len(arg_parts)))
      for arg_part, param_fn in zip(arg_parts, param_fns):
        param_elements.append(param_fn(arg_part))
    result_parts = wrapped_fn(*param_elements)

    # There is a tf.wrap_function(...) issue b/144127474 that variables created
    # from tf.import_graph_def(...) inside tf.wrap_function(...) is not
    # destroyed.  So get all the variables from `wrapped_fn` and destroy
    # manually.
    # TODO(b/144127474): Remove this manual cleanup once tf.wrap_function(...)
    # is fixed.
    resources = []
    for op in wrapped_fn.graph.get_operations():
      if op.type == 'VarHandleOp':
        resources += op.outputs
    if resources:
      for resource in wrapped_fn.prune(feeds={}, fetches=resources)():
        tf.raw_ops.DestroyResourceOp(resource=resource)

    result_elements = []
    for result_part, result_fn in zip(result_parts, result_fns):
      result_elements.append(result_fn(result_part))
    return anonymous_tuple.pack_sequence_as(result_type, result_elements)

  fn_to_return = lambda arg, p=param_fns, w=wrapped_fn: _fn_to_return(arg, p, w)

  # pylint: disable=function-redefined
  if must_pin_function_to_cpu:
    old_fn_to_return = fn_to_return

    def fn_to_return(x):
      with tf.device('cpu'):
        return old_fn_to_return(x)
  elif device is not None:
    old_fn_to_return = fn_to_return

    def fn_to_return(x):
      with tf.device(device.name):
        return old_fn_to_return(x)

  # pylint: enable=function-redefined

  if param_type is not None:
    return lambda arg: fn_to_return(arg)  # pylint: disable=unnecessary-lambda
  else:
    return lambda: fn_to_return(None)


def to_representation_for_type(
    value: Any,
    tf_function_cache: MutableMapping[str, Any],
    type_spec: Optional[computation_types.Type] = None,
    device: Optional[tf.config.LogicalDevice] = None) -> Any:
  """Verifies or converts the `value` to an eager object matching `type_spec`.

  WARNING: This function is only partially implemented. It does not support
  data sets at this point.

  The output of this function is always an eager tensor, eager dataset, a
  representation of a TensorFlow computation, or a nested structure of those
  that matches `type_spec`, and when `device` has been specified, everything
  is placed on that device on a best-effort basis.

  TensorFlow computations are represented here as zero- or one-argument Python
  callables that accept their entire argument bundle as a single Python object.

  Args:
    value: The raw representation of a value to compare against `type_spec` and
      potentially to be converted.
    tf_function_cache: A cache obeying `dict` semantics that can be used to look
      up previously embedded TensorFlow functions.
    type_spec: An instance of `tff.Type`, can be `None` for values that derive
      from `typed_object.TypedObject`.
    device: An optional `tf.config.LogicalDevice` to place the value on (for
      tensor-level values).

  Returns:
    Either `value` itself, or a modified version of it.

  Raises:
    TypeError: If the `value` is not compatible with `type_spec`.
  """
  type_spec = type_utils.reconcile_value_with_type_spec(value, type_spec)
  if isinstance(value, computation_base.Computation):
    return to_representation_for_type(
        computation_impl.ComputationImpl.get_proto(value), tf_function_cache,
        type_spec, device)
  elif isinstance(value, pb.Computation):
    key = (value.SerializeToString(), str(type_spec),
           device.name if device else None)
    cached_fn = tf_function_cache.get(key)
    if cached_fn is not None:
      return cached_fn
    embedded_fn = embed_tensorflow_computation(value, type_spec, device)
    tf_function_cache[key] = embedded_fn
    return embedded_fn
  elif type_spec.is_tuple():
    type_elem = anonymous_tuple.to_elements(type_spec)
    value_elem = (
        anonymous_tuple.to_elements(anonymous_tuple.from_container(value)))
    result_elem = []
    if len(type_elem) != len(value_elem):
      raise TypeError('Expected a {}-element tuple, found {} elements.'.format(
          len(type_elem), len(value_elem)))
    for (t_name, el_type), (v_name, el_val) in zip(type_elem, value_elem):
      if t_name != v_name:
        raise TypeError(
            'Mismatching element names in type vs. value: {} vs. {}.'.format(
                t_name, v_name))
      el_repr = to_representation_for_type(el_val, tf_function_cache, el_type,
                                           device)
      result_elem.append((t_name, el_repr))
    return anonymous_tuple.AnonymousTuple(result_elem)
  elif device is not None:
    py_typecheck.check_type(device, tf.config.LogicalDevice)
    with tf.device(device.name):
      return to_representation_for_type(
          value, tf_function_cache, type_spec=type_spec, device=None)
  elif isinstance(value, EagerValue):
    return value.internal_representation
  elif isinstance(value, executor_value_base.ExecutorValue):
    raise TypeError(
        'Cannot accept a value embedded within a non-eager executor.')
  elif type_spec.is_tensor():
    if not tf.is_tensor(value):
      value = tf.convert_to_tensor(value, dtype=type_spec.dtype)
    elif hasattr(value, 'read_value'):
      # a tf.Variable-like result, get a proper tensor.
      value = value.read_value()
    value_type = (
        computation_types.TensorType(value.dtype.base_dtype, value.shape))
    if not type_spec.is_assignable_from(value_type):
      raise TypeError(
          'The apparent type {} of a tensor {} does not match the expected '
          'type {}.'.format(value_type, value, type_spec))
    return value
  elif type_spec.is_sequence():
    if isinstance(value, list):
      value = tensorflow_utils.make_data_set_from_elements(
          None, value, type_spec.element)
    py_typecheck.check_type(value,
                            type_conversions.TF_DATASET_REPRESENTATION_TYPES)
    element_type = computation_types.to_type(value.element_spec)
    value_type = computation_types.SequenceType(element_type)
    type_spec.check_assignable_from(value_type)
    return value
  else:
    raise TypeError('Unexpected type {}.'.format(type_spec))


class EagerValue(executor_value_base.ExecutorValue):
  """A representation of an eager value managed by the eager executor."""

  def __init__(self, value, tf_function_cache, type_spec=None, device=None):
    """Creates an instance of a value in this executor.

    Args:
      value: Depending on `type_spec`, either a `tf.Tensor`, `tf.data.Dataset`,
        or a nested structure of these stored in an `AnonymousTuple`.
      tf_function_cache: A cache obeying `dict` semantics that can be used to
        look up previously embedded TensorFlow functions.
      type_spec: An instance of `tff.Type` that represents a tensor, a dataset,
        or a nested structure of these.
      device: An optional `tf.config.LogicalDevice` on which to place the value.
    """
    if type_spec is None:
      py_typecheck.check_type(value, typed_object.TypedObject)
      type_spec = value.type_signature
    else:
      type_spec = computation_types.to_type(type_spec)
      py_typecheck.check_type(type_spec, computation_types.Type)
    self._type_signature = type_spec
    self._value = to_representation_for_type(value, tf_function_cache,
                                             type_spec, device)

  @property
  def internal_representation(self):
    """Returns a representation of the eager value embedded in the executor.

    This property is only intended for use by the eager executor and tests. Not
    for consumption by consumers of the executor interface.
    """
    return self._value

  @property
  def type_signature(self):
    return self._type_signature

  @tracing.trace
  async def compute(self):
    return self._value


class EagerTFExecutor(executor_base.Executor):
  """The eager executor only runs TensorFlow, synchronously, in eager mode.

  TODO(b/134764569): Add support for data as a building block.

  This executor understands the following TFF types: tensors, sequences, named
  tuples, and functions. It does not understand placements, federated, or
  abstract types.

  This executor understands the following kinds of TFF computation building
  blocks: tensorflow computations, and external data. It does not understand
  lambda calculus or any compositional constructs. Tuples and selections can
  only be created using `create_tuple()` and `create_selection()` in the API.

  The arguments to be ingested can be Python constants of simple types, nested
  structures of those, as well as eager tensors and eager datasets.

  The external data references must identify files available in the executor's
  filesystem. The exact format is yet to be documented.

  The executor will be able to place work on specific devices (e.g., on GPUs).
  In contrast to the reference executor, it handles data sets in a pipelined
  fashion, and does not place limits on the data set sizes. It also avoids
  marshaling TensorFlow values in and out between calls.

  It does not deal with multithreading, checkpointing, federated computations,
  and other concerns to be covered by separate executor components. It runs the
  operations it supports in a synchronous fashion. Asynchrony and other aspects
  not supported here should be handled by composing this executor with other
  executors into a complex executor stack, rather than mixing in all the logic.
  """

  def __init__(self, device=None):
    """Creates a new instance of an eager executor.

    Args:
      device: An optional `tf.config.LogicalDevice` that this executor will
        schedule all of its operations to run on. For example, the list of
        logical devices can be obtained using
        `tf.config.list_logical_devices()`.

    Raises:
      RuntimeError: If not executing eagerly.
      TypeError: If the device is not a `tf.config.LogicalDevice`.
      ValueError: If there is no device `device`.
    """
    if not tf.executing_eagerly():
      raise RuntimeError('The eager executor may only be used in eager mode.')
    if device is not None:
      py_typecheck.check_type(device, tf.config.LogicalDevice)
      self._device = device
    else:
      self._device = None
    self._tf_function_cache = cachetools.LRUCache(_TF_FUNCTION_CACHE_SIZE)

  @tracing.trace(span=True)
  async def create_value(self, value, type_spec=None):
    """Embeds `value` of type `type_spec` within this executor.

    Args:
      value: An object that represents the value to embed within the executor.
      type_spec: The `tff.Type` of the value represented by this object, or
        something convertible to it. Can optionally be `None` if `value` is an
        instance of `typed_object.TypedObject`.

    Returns:
      An instance of `EagerValue`.

    Raises:
      RuntimeError: If not executing eagerly.
      TypeError: If the arguments are of the wrong types.
      ValueError: If the type was not specified and cannot be determined from
        the value.
    """
    if not tf.executing_eagerly():
      raise RuntimeError('The eager executor may only be used in eager mode.')

    return EagerValue(value, self._tf_function_cache, type_spec, self._device)

  @tracing.trace
  async def create_call(self, comp, arg=None):
    """Creates a call to `comp` with optional `arg`.

    Args:
      comp: As documented in `executor_base.Executor`.
      arg: As documented in `executor_base.Executor`.

    Returns:
      An instance of `EagerValue` representing the result of the call.

    Raises:
      RuntimeError: If not executing eagerly.
      TypeError: If the arguments are of the wrong types.
    """
    py_typecheck.check_type(comp, EagerValue)
    if arg is not None:
      py_typecheck.check_type(arg, EagerValue)
    if not comp.type_signature.is_function():
      raise TypeError('Expected a functional type, found {}'.format(
          comp.type_signature))
    if comp.type_signature.parameter is not None:
      return EagerValue(
          comp.internal_representation(arg.internal_representation),  # pytype: disable=attribute-error
          self._tf_function_cache,
          comp.type_signature.result,
          self._device)
    elif arg is None:
      return EagerValue(comp.internal_representation(), self._tf_function_cache,
                        comp.type_signature.result, self._device)
    else:
      raise TypeError('Cannot pass an argument to a no-argument function.')

  @tracing.trace
  async def create_tuple(self, elements):
    """Creates a tuple of `elements`.

    Args:
      elements: As documented in `executor_base.Executor`.

    Returns:
      An instance of `EagerValue` that represents the constructed tuple.
    """
    elements = anonymous_tuple.to_elements(
        anonymous_tuple.from_container(elements))
    val_elements = []
    type_elements = []
    for k, v in elements:
      py_typecheck.check_type(v, EagerValue)
      val_elements.append((k, v.internal_representation))
      type_elements.append((k, v.type_signature))
    return EagerValue(
        anonymous_tuple.AnonymousTuple(val_elements), self._tf_function_cache,
        computation_types.NamedTupleType([
            (k, v) if k is not None else v for k, v in type_elements
        ]))

  @tracing.trace
  async def create_selection(self, source, index=None, name=None):
    """Creates a selection from `source`.

    Args:
      source: As documented in `executor_base.Executor`.
      index: As documented in `executor_base.Executor`.
      name: As documented in `executor_base.Executor`.

    Returns:
      An instance of `EagerValue` that represents the constructed selection.

    Raises:
      TypeError: If arguments are of the wrong types.
      ValueError: If either both, or neither of `name` and `index` are present.
    """
    py_typecheck.check_type(source, EagerValue)
    py_typecheck.check_type(source.type_signature,
                            computation_types.NamedTupleType)
    py_typecheck.check_type(source.internal_representation,
                            anonymous_tuple.AnonymousTuple)
    if index is not None:
      py_typecheck.check_type(index, int)
      if name is not None:
        raise ValueError(
            'Cannot simultaneously specify name {} and index {}.'.format(
                name, index))
      else:
        return EagerValue(source.internal_representation[index],
                          self._tf_function_cache, source.type_signature[index])
    elif name is not None:
      py_typecheck.check_type(name, str)
      return EagerValue(
          getattr(source.internal_representation, str(name)),
          self._tf_function_cache, getattr(source.type_signature, str(name)))
    else:
      raise ValueError('Must specify either name or index.')

  def close(self):
    pass

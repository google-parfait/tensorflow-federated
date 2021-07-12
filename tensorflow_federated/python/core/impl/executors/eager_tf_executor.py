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

import itertools
from typing import Any, Iterable, MutableMapping, Optional

from absl import logging
import cachetools
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.types import typed_object
from tensorflow_federated.python.core.impl.utils import tensorflow_utils
from tensorflow_federated.python.tensorflow_libs import graph_merge

# Cache size here is simply heuristic, no formal analysis.
_TF_FUNCTION_CACHE_SIZE = 100


def _all_graph_def_nodes(
    graph_def: tf.compat.v1.GraphDef) -> Iterable[tf.compat.v1.NodeDef]:
  return itertools.chain(graph_def.node,
                         *[f.node_def for f in graph_def.library.function])


# TODO(b/159180073): remove this check after enabling reduce op for multi-GPU
def _check_dataset_reduce_for_multi_gpu(
    graph_def: tf.compat.v1.GraphDef) -> None:
  """Detect if ReduceDataset Op is used in a multi-GPU simulation."""
  has_dataset_reduce_node = False
  for node in _all_graph_def_nodes(graph_def):
    # If `tf.device` is explicitly used in the graph_def, the graph_def was
    # defined by advanced users who we trust know what they are doing.
    if node.device:
      return
    if node.op == 'ReduceDataset':
      has_dataset_reduce_node = True
  if has_dataset_reduce_node:
    raise ValueError(
        'Detected dataset reduce op in multi-GPU TFF simulation: '
        '`use_experimental_simulation_loop=True` for `tff.learning`; or use '
        '`for ... in iter(dataset)` for your own dataset iterations. See '
        'https://www.tensorflow.org/federated/tutorials/simulations_with_accelerators'
        ' for examples.')


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


def _call_embedded_tf(*, arg, param_fns, result_fns, result_type, wrapped_fn,
                      destroy_before_invocation, destroy_after_invocation):
  """Function to be run upon EagerTFExecutor.create_call invocation.

  As this function is run completely synchronously, and
  `EagerTFExecutor.create_call` invocations represent the main work of the
  program, this function should be kept as-thin a wrapper around delegation
  to the eager TensorFlow runtime as possible.

  Args:
    arg: Argument on which to invoke embedded function.
    param_fns: Functions to be applied to elements of `arg` before passing to
      `wrapped_fn`, to prepare these argument for ingestion by the eager TF
      runtime.
    result_fns: Functions to be applied to results of calling `wrapped_fn` on
      arg before re-embedding as EagerTFExecutor values.
    result_type: TFF Type signature of the result of `wrapped_fn`.
    wrapped_fn: Result of `tf.compat.v1.wrap_function` to run in the eager TF
      runtime.
    destroy_before_invocation: eager TF runtime resources which should be
      destroyed before invoking `wrapped_fn`. Examples might include hashtable
      resources.
    destroy_after_invocation: eager TF runtime resources which should be
      destroyed after invoking `wrapped_fn`. Examples might include resource
      variables.

  Returns:
    A `structure.Struct` representing the result of invoking `wrapped_fn` on
    `arg`. The result of invoking `wrapped_fn` on `arg`, postprocessed by
    `result_fns` and packed as `result_type`. This result must be structured
    such that `to_representation_for_type` with this result and `result_type` as
    an argument would no-op.

  Raises:
    RuntimeError: If `arg` and `param_fns` have different numbers of elements.
  """

  # TODO(b/166479382): This cleanup-before-invocation pattern is a workaround
  # to square the circle of TF data expecting to lazily reference this
  # resource on iteration, as well as usages that expect to reinitialize a
  # table with new data. Revisit the semantics implied by this cleanup
  # pattern.
  with tracing.span(
      'EagerTFExecutor.create_call',
      'resource_cleanup_before_invocation',
      span=True):
    for resource in destroy_before_invocation:
      tf.raw_ops.DestroyResourceOp(resource=resource)

  param_elements = []
  if arg is not None:
    with tracing.span(
        'EagerTFExecutor.create_call', 'arg_ingestion', span=True):
      arg_parts = structure.flatten(arg)
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
  with tracing.span(
      'EagerTFExecutor.create_call',
      'resource_cleanup_after_invocation',
      span=True):
    for resource in destroy_after_invocation:
      tf.raw_ops.DestroyResourceOp(resource=resource)

  with tracing.span('EagerTFExecutor.create_call', 'result_packing', span=True):
    result_elements = []
    for result_part, result_fn in zip(result_parts, result_fns):
      result_elements.append(result_fn(result_part))
    return structure.pack_sequence_as(result_type, result_elements)


def _ensure_comp_runtime_compatible(comp: pb.Computation) -> pb.Computation:
  """Ensures `comp` is compatible with eager runtime backing EagerExecutor."""
  original_tf = comp.tensorflow
  graph_def = serialization_utils.unpack_graph_def(original_tf.graph_def)
  # TODO(b/159180073): clean raise after fixing dataset reduce.
  num_gpu_devices = len(tf.config.list_logical_devices('GPU'))
  if num_gpu_devices > 1:
    _check_dataset_reduce_for_multi_gpu(graph_def)

  return comp


@tracing.trace
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
  comp = _ensure_comp_runtime_compatible(comp)
  comp_type = type_serialization.deserialize_type(comp.type)
  type_spec = computation_types.to_type(type_spec)
  if type_spec is not None:
    if not type_spec.is_equivalent_to(comp_type):
      raise TypeError('Expected a computation of type {}, got {}.'.format(
          type_spec, comp_type))
  else:
    type_spec = comp_type
  # TODO(b/156302055): Currently, TF will raise on any function returning a
  # `tf.data.Dataset` not pinned to CPU. We should follow up here and remove
  # this gating when we can.
  must_pin_function_to_cpu = type_analysis.contains(type_spec.result,
                                                    lambda t: t.is_sequence())
  which_computation = comp.WhichOneof('computation')
  if which_computation != 'tensorflow':
    unexpected_building_block = building_blocks.ComputationBuildingBlock.from_proto(
        comp)
    raise TypeError('Expected a TensorFlow computation, found {}.'.format(
        unexpected_building_block))

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
    for spec in structure.flatten(type_spec.parameter):
      if spec.is_tensor():
        param_fns.append(lambda x: x)
      else:
        py_typecheck.check_type(spec, computation_types.SequenceType)
        param_fns.append(tf.data.experimental.to_variant)

  result_fns = []
  for spec in structure.flatten(result_type):
    if spec.is_tensor():
      result_fns.append(lambda x: x)
    else:
      py_typecheck.check_type(spec, computation_types.SequenceType)
      tf_structure = type_conversions.type_to_tf_structure(spec.element)

      def fn(x, tf_structure=tf_structure):
        return tf.data.experimental.from_variant(x, tf_structure)

      result_fns.append(fn)

  ops = wrapped_fn.graph.get_operations()

  eager_cleanup_ops = []
  destroy_before_invocation = []
  for op in ops:
    if op.type == 'HashTableV2':
      eager_cleanup_ops += op.outputs
  if eager_cleanup_ops:
    for resource in wrapped_fn.prune(feeds={}, fetches=eager_cleanup_ops)():
      destroy_before_invocation.append(resource)

  lazy_cleanup_ops = []
  destroy_after_invocation = []
  for op in ops:
    if op.type == 'VarHandleOp':
      lazy_cleanup_ops += op.outputs
  if lazy_cleanup_ops:
    for resource in wrapped_fn.prune(feeds={}, fetches=lazy_cleanup_ops)():
      destroy_after_invocation.append(resource)

  def fn_to_return(arg,
                   param_fns=tuple(param_fns),
                   result_fns=tuple(result_fns),
                   result_type=result_type,
                   wrapped_fn=wrapped_fn,
                   destroy_before=tuple(destroy_before_invocation),
                   destroy_after=tuple(destroy_after_invocation)):
    # This double-function pattern works around python late binding, forcing the
    # variables to bind eagerly.
    return _call_embedded_tf(
        arg=arg,
        param_fns=param_fns,
        result_fns=result_fns,
        result_type=result_type,
        wrapped_fn=wrapped_fn,
        destroy_before_invocation=destroy_before,
        destroy_after_invocation=destroy_after)

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


@tracing.trace
def _to_computation_internal_rep(*, value: pb.Computation,
                                 tf_function_cache: MutableMapping[str, Any],
                                 type_spec: computation_types.StructType,
                                 device: tf.config.LogicalDevice):
  """Converts a `pb.Computation` to a `tf.function`."""
  if value.tensorflow.cache_key.id:
    logging.debug('Using value id for cache key: %s',
                  value.tensorflow.cache_key.id)
    key = (value.tensorflow.cache_key.id,
           type_serialization.serialize_type(type_spec).SerializeToString(
               deterministic=True), device.name if device else None)
  else:
    logging.debug('Using hash of graph_def for cache key')
    key = (value.SerializeToString(deterministic=True),
           type_serialization.serialize_type(type_spec).SerializeToString(
               deterministic=True), device.name if device else None)
  cached_fn = tf_function_cache.get(key)
  if cached_fn is not None:
    return cached_fn
  embedded_fn = embed_tensorflow_computation(value, type_spec, device)
  tf_function_cache[key] = embedded_fn
  return embedded_fn


@tracing.trace
def _to_struct_internal_rep(
    *, value: Any, tf_function_cache: MutableMapping[str, Any],
    type_spec: computation_types.StructType,
    device: tf.config.LogicalDevice) -> structure.Struct:
  """Converts a python container to internal representation for TF executor."""
  type_iterator = structure.iter_elements(type_spec)
  value_struct = structure.from_container(value)
  value_iterator = structure.iter_elements(value_struct)

  if len(type_spec) != len(value_struct):
    raise TypeError('Mismatched number of elements between type spec and value '
                    'in `to_representation_for_type`. Type spec has {} '
                    'elements, value has {}.'.format(
                        len(type_spec), len(value_struct)))
  result_elem = []
  for (type_name, elem_type), (val_name,
                               elem_val) in zip(type_iterator, value_iterator):
    if val_name is not None and type_name != val_name:
      raise TypeError(
          'Mismatching element names in type vs. value: {} vs. {}.'.format(
              type_name, val_name))
    elem_repr = to_representation_for_type(elem_val, tf_function_cache,
                                           elem_type, device)
    result_elem.append((type_name, elem_repr))
  return structure.Struct(result_elem)


@tracing.trace
def _to_tensor_internal_rep(*, value: Any,
                            type_spec: computation_types.Type) -> tf.Tensor:
  """Normalizes tensor-like value to a tf.Tensor."""
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


@tracing.trace
def _to_sequence_internal_rep(
    *, value: Any, type_spec: computation_types.Type) -> tf.data.Dataset:
  """Ingests `value`, converting to an eager dataset."""
  if isinstance(value, list):
    value = tensorflow_utils.make_data_set_from_elements(
        None, value, type_spec.element)
  if isinstance(value, type_conversions.TF_DATASET_REPRESENTATION_TYPES):
    element_type = computation_types.to_type(value.element_spec)
    value_type = computation_types.SequenceType(element_type)
    type_spec.check_assignable_from(value_type)
    return value
  py_typecheck.check_type(type_spec, computation_types.SequenceType)
  output_sig = type_conversions.type_to_tf_tensor_specs(type_spec.element)
  return tf.data.Dataset.from_generator(value, output_signature=output_sig)


@tracing.trace
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
  type_spec = executor_utils.reconcile_value_with_type_spec(value, type_spec)
  if isinstance(value, computation_base.Computation):
    return to_representation_for_type(
        computation_impl.ComputationImpl.get_proto(value), tf_function_cache,
        type_spec, device)
  elif isinstance(value, pb.Computation):
    return _to_computation_internal_rep(
        value=value,
        tf_function_cache=tf_function_cache,
        type_spec=type_spec,
        device=device)
  elif type_spec.is_struct():
    return _to_struct_internal_rep(
        value=value,
        tf_function_cache=tf_function_cache,
        type_spec=type_spec,
        device=device)
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
    return _to_tensor_internal_rep(value=value, type_spec=type_spec)
  elif type_spec.is_sequence():
    return _to_sequence_internal_rep(value=value, type_spec=type_spec)
  else:
    raise TypeError(
        f'Unexpected type {type_spec} for value of type {type(value)}: {value}')


class EagerValue(executor_value_base.ExecutorValue):
  """A representation of an eager value managed by the eager executor."""

  def __init__(self, value, type_spec):
    """Creates an instance of a value in this executor.

    Args:
      value: Depending on `type_spec`, either a `tf.Tensor`, `tf.data.Dataset`,
        or a nested structure of these stored in an `Struct`. We assume that
        `value` is a fixed point of `to_representation_for_type` with type_spec
        argument `type_spec`.
      type_spec: An instance of `tff.Type` that represents a tensor, a dataset,
        or a nested structure of these.
    """
    py_typecheck.check_type(type_spec, computation_types.Type)
    self._type_signature = type_spec
    self._value = value

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
  only be created using `create_struct()` and `create_selection()` in the API.

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

  One further implementation detail is worth noting. Like all executors, this
  executor embeds incoming data as an instance of an executor-specific class,
  here the `EagerValue`. All `EagerValues` are assumed in this implmentation
  to have an `internal_representation` which is a fixed point under the action
  of `to_representation_for_type` with type the `type_signature` attribute of
  the `EagerValue`. This invariant is introduced by normalization in
  `create_value`, and is respected by the form of returned `EagerValues` in all
  other methods this executor exposes.
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

    `create_value` first normalizes its incoming `value` arguments via
    `to_representation_for_type`, establishing the invariant that every value
    embedded in this executor would no-op under the action of
    `to_representation_for_type`. This invariant is then preserved and assumed
    by the remainder of the methods exposed by the executor.

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

    if type_spec is None:
      py_typecheck.check_type(value, typed_object.TypedObject)
      type_spec = value.type_signature
    else:
      type_spec = computation_types.to_type(type_spec)
      py_typecheck.check_type(type_spec, computation_types.Type)
    normalized_value = to_representation_for_type(value,
                                                  self._tf_function_cache,
                                                  type_spec, self._device)
    return EagerValue(normalized_value, type_spec)

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
          comp.internal_representation(arg.internal_representation),
          comp.type_signature.result)
    elif arg is None:
      return EagerValue(comp.internal_representation(),
                        comp.type_signature.result)
    else:
      raise TypeError('Cannot pass an argument to a no-argument function.')

  @tracing.trace
  async def create_struct(self, elements):
    """Creates a tuple of `elements`.

    Args:
      elements: As documented in `executor_base.Executor`.

    Returns:
      An instance of `EagerValue` that represents the constructed tuple.
    """
    elements = structure.iter_elements(structure.from_container(elements))
    val_elements = []
    type_elements = []
    for k, v in elements:
      py_typecheck.check_type(v, EagerValue)
      val_elements.append((k, v.internal_representation))
      type_elements.append((k, v.type_signature))
    return EagerValue(
        structure.Struct(val_elements),
        computation_types.StructType([
            (k, v) if k is not None else v for k, v in type_elements
        ]))

  @tracing.trace
  async def create_selection(self, source, index):
    """Creates a selection from `source`.

    Args:
      source: As documented in `executor_base.Executor`.
      index: As documented in `executor_base.Executor`.

    Returns:
      An instance of `EagerValue` that represents the constructed selection.

    Raises:
      TypeError: If arguments are of the wrong types.
      ValueError: If either both, or neither of `name` and `index` are present.
    """
    py_typecheck.check_type(source, EagerValue)
    py_typecheck.check_type(source.type_signature, computation_types.StructType)
    py_typecheck.check_type(source.internal_representation, structure.Struct)
    py_typecheck.check_type(index, int)
    return EagerValue(source.internal_representation[index],
                      source.type_signature[index])

  def close(self):
    pass

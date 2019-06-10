# Lint as: python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import typed_object
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import executor_value_base
from tensorflow_federated.python.core.impl import graph_utils
from tensorflow_federated.python.core.impl import type_serialization
from tensorflow_federated.python.core.impl import type_utils

_AVAILABLE_DEVICES = list(
    '/{}'.format(str(x.name[17:]))
    for x in tf.config.experimental.list_physical_devices()
    if x.name.startswith('/physical_device:'))


def get_available_devices():
  return _AVAILABLE_DEVICES


def embed_tensorflow_computation(comp, type_spec=None, device=None):
  """Embeds a TensorFlow computation for use in the eager context.

  Args:
    comp: An instance of `pb.Computation`.
    type_spec: An optional `tff.Type` instance or something convertible to it.
    device: An optional device name.

  Returns:
    Either a one-argument or a zero-argument callable that executes the
    computation in eager mode.

  Raises:
    TypeError: If arguments are of the wrong types, e.g., in `comp` is not a
      TensorFlow computation.
  """
  # TODO(b/134543154): Decide whether this belongs in `graph_utils.py` since
  # it deals exclusively with eager mode. Incubate here, and potentially move
  # there, once stable.

  if device is not None:
    raise NotImplementedError('Unable to embed TF code on a specific device.')

  py_typecheck.check_type(comp, pb.Computation)
  comp_type = type_serialization.deserialize_type(comp.type)
  type_spec = computation_types.to_type(type_spec)
  if type_spec is not None:
    if not type_utils.are_equivalent_types(type_spec, comp_type):
      raise TypeError('Expected a computation of type {}, got {}.'.format(
          str(type_spec), str(comp_type)))
  else:
    type_spec = comp_type
  which_computation = comp.WhichOneof('computation')
  if which_computation != 'tensorflow':
    raise TypeError('Expected a TensorFlow computation, found {}.'.format(
        which_computation))

  if isinstance(type_spec, computation_types.FunctionType):
    param_type = type_spec.parameter
    result_type = type_spec.result
  else:
    param_type = None
    result_type = type_spec

  if param_type is not None:
    input_tensor_names = graph_utils.extract_tensor_names_from_binding(
        comp.tensorflow.parameter)
  else:
    input_tensor_names = []

  output_tensor_names = graph_utils.extract_tensor_names_from_binding(
      comp.tensorflow.result)

  def function_to_wrap(*args):
    if len(args) != len(input_tensor_names):
      raise RuntimeError('Expected {} arguments, found {}.'.format(
          str(len(input_tensor_names)), str(len(args))))
    graph_def = serialization_utils.unpack_graph_def(comp.tensorflow.graph_def)
    return tf.import_graph_def(
        graph_def,
        input_map=dict(zip(input_tensor_names, args)),
        return_elements=output_tensor_names)

  signature = []
  param_fns = []
  if param_type is not None:
    for spec in anonymous_tuple.flatten(type_spec.parameter):
      if isinstance(spec, computation_types.TensorType):
        signature.append(tf.TensorSpec(spec.shape, spec.dtype))
        param_fns.append(lambda x: x)
      else:
        py_typecheck.check_type(spec, computation_types.SequenceType)
        signature.append(tf.TensorSpec([], tf.variant))
        param_fns.append(tf.data.experimental.to_variant)

  wrapped_fn = tf.compat.v1.wrap_function(function_to_wrap, signature)

  result_fns = []
  for spec in anonymous_tuple.flatten(result_type):
    if isinstance(spec, computation_types.TensorType):
      result_fns.append(lambda x: x)
    else:
      py_typecheck.check_type(spec, computation_types.SequenceType)
      structure = type_utils.type_to_tf_structure(spec.element)

      def fn(x, structure=structure):
        return tf.data.experimental.from_variant(x, structure)

      result_fns.append(fn)

  def _fn_to_return(arg, param_fns, wrapped_fn):  # pylint:disable=missing-docstring
    param_elements = []
    if arg is not None:
      arg_parts = anonymous_tuple.flatten(arg)
      if len(arg_parts) != len(param_fns):
        raise RuntimeError('Expected {} arguments, found {}.'.format(
            str(len(param_fns)), str(len(arg_parts))))
      for arg_part, param_fn in zip(arg_parts, param_fns):
        param_elements.append(param_fn(arg_part))
    result_parts = wrapped_fn(*param_elements)
    result_elements = []
    for result_part, result_fn in zip(result_parts, result_fns):
      result_elements.append(result_fn(result_part))
    return anonymous_tuple.pack_sequence_as(result_type, result_elements)

  fn_to_return = lambda arg, p=param_fns, w=wrapped_fn: _fn_to_return(arg, p, w)
  if param_type is not None:
    return lambda arg: fn_to_return(arg)  # pylint: disable=unnecessary-lambda
  else:
    return lambda: fn_to_return(None)


def to_representation_for_type(value, type_spec=None, device=None):
  """Verifies or converts the `value` to an eager objct matching `type_spec`.

  WARNING: This function is only partially implemented. It does not support
  data sets at this point.

  The output of this function is always an eager tensor, eager dataset, a
  representation of a TensorFlow computtion, or a nested structure of those
  that matches `type_spec`, and when `device` has been specified, everything
  is placed on that device on a best-effort basis.

  TensorFlow computations are represented here as zero- or one-argument Python
  callables that accept their entire argument bundle as a single Python object.

  Args:
    value: The raw representation of a value to compare against `type_spec` and
      potentially to be converted.
    type_spec: An instance of `tff.Type`, can be `None` for values that derive
      from `typed_object.TypedObject`.
    device: The optional device to place the value on (for tensor-level values).

  Returns:
    Either `value` itself, or a modified version of it.

  Raises:
    TypeError: If the `value` is not compatible with `type_spec`.
  """
  if device is not None:
    py_typecheck.check_type(device, six.string_types)
    with tf.device(device):
      return to_representation_for_type(value, type_spec=type_spec, device=None)
  type_spec = computation_types.to_type(type_spec)
  if isinstance(value, typed_object.TypedObject):
    if type_spec is not None:
      if not type_utils.are_equivalent_types(value.type_signature, type_spec):
        raise TypeError('Expected a value of type {}, found {}.'.format(
            str(type_spec), str(value.type_signature)))
    else:
      type_spec = value.type_signature
  if type_spec is None:
    raise ValueError(
        'Cannot derive an eager representation for a value of an unknown type.')
  if isinstance(value, EagerValue):
    return value.internal_representation
  if isinstance(value, executor_value_base.ExecutorValue):
    raise TypeError(
        'Cannot accept a value embedded within a non-eager executor.')
  if isinstance(value, computation_base.Computation):
    return to_representation_for_type(
        computation_impl.ComputationImpl.get_proto(value), type_spec, device)
  if isinstance(value, pb.Computation):
    return embed_tensorflow_computation(value, type_spec, device)
  if isinstance(type_spec, computation_types.TensorType):
    if not isinstance(value, tf.Tensor):
      value = tf.constant(value, type_spec.dtype, type_spec.shape)
    value_type = (
        computation_types.TensorType(value.dtype.base_dtype, value.shape))
    if not type_utils.is_assignable_from(type_spec, value_type):
      raise TypeError(
          'The apparent type {} of a tensor {} does not match the expected '
          'type {}.'.format(str(value_type), str(value), str(type_spec)))
    return value
  elif isinstance(type_spec, computation_types.NamedTupleType):
    type_elem = anonymous_tuple.to_elements(type_spec)
    value_elem = (
        anonymous_tuple.to_elements(anonymous_tuple.from_container(value)))
    result_elem = []
    if len(type_elem) != len(value_elem):
      raise TypeError('Expected a {}-element tuple, found {} elements.'.format(
          str(len(type_elem)), str(len(value_elem))))
    for (t_name, el_type), (v_name, el_val) in zip(type_elem, value_elem):
      if t_name != v_name:
        raise TypeError(
            'Mismatching element names in type vs. value: {} vs. {}.'.format(
                t_name, v_name))
      el_repr = to_representation_for_type(el_val, el_type, device)
      result_elem.append((t_name, el_repr))
    return anonymous_tuple.AnonymousTuple(result_elem)
  else:
    raise TypeError('Unexpected type {}.'.format(str(type_spec)))


class EagerValue(executor_value_base.ExecutorValue):
  """A representation of an eager value managed by the eager executor."""

  def __init__(self, value, type_spec, device=None):
    """Creates an instance of a value in this executor.

    Args:
      value: Depending on `type_spec`, either a `tf.Tensor`, `tf.data.Dataset`,
        or a nested structure of these stored in an `AnonymousTuple`.
      type_spec: An instance of `tff.Type` that represents a tensor, a dataset,
        or a nested structure of these.
      device: The optional device on which to place the value.
    """
    type_spec = computation_types.to_type(type_spec)
    self._value = to_representation_for_type(value, type_spec, device)
    if type_spec is None:
      py_typecheck.check_type(self._value, typed_object.TypedObject)
      type_spec = self._value.type_signature
    else:
      py_typecheck.check_type(type_spec, computation_types.Type)
    self._type_signature = type_spec

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


class EagerExecutor(executor_base.Executor):
  """The eager executor only runs TensorFlow, synchronously, in eager mode.

  WARNING: This executor is only partially implemented, and should not be used.

  This executor understands tensors, datasets, nested structures of those, and
  tensorflow computations. It will optionally be able to place work on specific
  devices (e.g., GPUs). In contrast to the reference executor, it handles data
  sets in a pipelined fashion, and does not place limits on the data set sizes.

  It does not deal with multithreading, checkpointing, federated computations,
  and other concerns to be covered by separate executor components. It runs the
  operations it supports in a synchronous fashion.
  """

  def __init__(self, device=None):
    """Creates a new instance of an eager executor.

    Args:
      device: An optional name of the device that this executor will schedule
        all of its operations to run on. Must be one of the devices returned by
        `get_available_devices`.

    Raises:
      RuntimeError: If not executing eagerly.
      TypeError: If the device name is not a string.
      ValueError: If there is no device `device`.
    """
    if not tf.executing_eagerly():
      raise RuntimeError('The eager executor may only be used in eager mode.')
    if device is not None:
      py_typecheck.check_type(device, six.string_types)
      if device not in get_available_devices():
        raise ValueError('No device "{}"; available devices are {}'.format(
            device, str(get_available_devices())))
      self._device = device
    else:
      self._device = None

  def ingest(self, value, type_spec=None):
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
    return EagerValue(value, type_spec, self._device)

  def invoke(self, comp, arg):
    """Invokes `comp` on optional `arg` in the eager executor.

    Args:
      comp: As documented in `executor_base.Executor`.
      arg: As documented in `executor_base.Executor`.

    Returns:
      An instance of `EagerValue` representing the result of the invocation.

    Raises:
      RuntimeError: If not executing eagerly.
      TypeError: If the arguments are of the wrong types.
    """
    py_typecheck.check_type(comp, typed_object.TypedObject)
    if not isinstance(comp.type_signature, computation_types.FunctionType):
      raise TypeError('Expected a functional type, found {}'.format(
          str(comp.type_signature)))
    comp = self.ingest(comp, comp.type_signature)
    if comp.type_signature.parameter is not None:
      arg = self.ingest(arg, comp.type_signature.parameter)
      return EagerValue(
          comp.internal_representation(arg.internal_representation),
          comp.type_signature.result, self._device)
    elif arg is None:
      return EagerValue(comp.internal_representation(),
                        comp.type_signature.result, self._device)
    else:
      raise TypeError('Cannot pass an argument to a no-argument function.')

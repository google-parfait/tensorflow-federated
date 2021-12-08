# Copyright 2021, The TensorFlow Federated Authors.
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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""An executor responsible for operations on sequences."""

import abc
import asyncio

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.types import typed_object


class _AsyncFromSyncIterator(object):
  """Wraps a regular sync iterator as an asynchronous one."""

  def __init__(self, sync_iter):
    self._iter = sync_iter

  async def __anext__(self):
    try:
      return next(self._iter)
    except StopIteration:
      raise StopAsyncIteration


class _SyncFromAsyncIterable(object):
  """Wraps an asynchronous iteble as a regular sync iterable and generator."""

  def __init__(self, async_iterable):
    self._async_iterable = async_iterable
    self._event_loop = asyncio.new_event_loop()

  def __call__(self):
    return self.__iter__()

  def __iter__(self):

    def _generate(event_loop, async_iter):
      try:
        while True:
          # TODO(b/175888145): Whiel everything works so far with the full
          # stack on top of this, there is some lignering doubt about whether
          # asyncio will break here. If we see breakage, possibly reuse the
          # pattern from thread delegating executor, with a dedicated thread.
          yield event_loop.run_until_complete(async_iter.__anext__())
      except StopAsyncIteration:
        return

    return _generate(self._event_loop, self._async_iterable.__aiter__())


class _Sequence(typed_object.TypedObject, metaclass=abc.ABCMeta):
  """An internal representation of a sequence within this executor."""

  def __init__(self, type_spec: computation_types.SequenceType):
    py_typecheck.check_type(type_spec, computation_types.SequenceType)
    self._type_signature = type_spec

  @property
  def type_signature(self):
    return self._type_signature

  @abc.abstractmethod
  async def compute(self):
    raise NotImplementedError

  @abc.abstractmethod
  def __aiter__(self):
    raise NotImplementedError


class _SequenceFromPayload(_Sequence):
  """An internal representation of a sequence created from a payload."""

  def __init__(self, payload, type_spec: computation_types.SequenceType):
    """Constructs a representation for sequence-typed `payload`.

    Args:
      payload: The Python object that contains the payload.
      type_spec: An instance of `computation_types.SequenceType`.
    """
    _Sequence.__init__(self, type_spec)
    if isinstance(payload, type_conversions.TF_DATASET_REPRESENTATION_TYPES):
      self._payload = payload
      return
    try:
      iter(payload)
      self._payload = payload
    except Exception as err:
      raise NotImplementedError(
          'Unrecognized type of payload: {}: returned {} from iter()'.format(
              py_typecheck.type_string(type(payload)), str(err)))

  async def compute(self):
    return self._payload

  def __aiter__(self):
    return _AsyncFromSyncIterator(iter(self._payload))


class _SequenceFromMap(_Sequence):
  """An internal representation of a sequence created from a map op."""

  def __init__(self, source: _Sequence,
               map_fn: executor_value_base.ExecutorValue,
               target_executor: executor_base.Executor,
               type_spec: computation_types.SequenceType):
    """Constructs a representation of a mapped sequence.

    Args:
      source: The source sequence.
      map_fn: A mapping function embedded in the target executor.
      target_executor: The target executor.
      type_spec: The type of the mapped sequence.
    """
    py_typecheck.check_type(source, _Sequence)
    py_typecheck.check_type(map_fn, executor_value_base.ExecutorValue)
    py_typecheck.check_type(target_executor, executor_base.Executor)
    py_typecheck.check_type(type_spec, computation_types.SequenceType)
    _Sequence.__init__(self, type_spec)
    self._source = source
    self._map_fn = map_fn
    self._target_executor = target_executor

  async def compute(self):
    return _SyncFromAsyncIterable(self)

  def __aiter__(self):
    return _AsyncMapIterator(self._source, self._map_fn, self._target_executor)


class _AsyncMapIterator(object):
  """An async iterator based on a mapping function."""

  def __init__(self, source: _Sequence,
               map_fn: executor_value_base.ExecutorValue,
               target_executor: executor_base.Executor):
    self._source_aiter = source.__aiter__()
    self._source_element_type = source.type_signature.element
    self._map_fn = map_fn
    self._target_executor = target_executor

  async def __anext__(self):
    next_element = await self._source_aiter.__anext__()
    element_val = await _delegate(next_element, self._source_element_type,
                                  self._target_executor)
    mapped_val = await self._target_executor.create_call(
        self._map_fn, element_val)
    return await mapped_val.compute()


class _SequenceOp(typed_object.TypedObject, metaclass=abc.ABCMeta):
  """An internal representation of a sequence op used by this executor."""

  def __init__(self, type_spec: computation_types.SequenceType):
    self._type_signature = type_spec

  @property
  def type_signature(self):
    return self._type_signature

  @abc.abstractmethod
  async def execute(self, target_executor, arg):
    raise NotImplementedError


class _SequenceMapOp(_SequenceOp):
  """Implements a SEQUENCE_MAP intrinsic."""

  async def execute(self, target_executor: executor_base.Executor, arg):
    arg_type = self._type_signature.parameter
    seq, map_fn = await asyncio.gather(*[
        _to_sequence(arg[1]),
        _delegate(arg[0], arg_type[0], target_executor)
    ])
    result_type = self._type_signature.result
    return _SequenceFromMap(seq, map_fn, target_executor, result_type)


class _SequenceReduceOp(_SequenceOp):
  """Implements a SEQUENCE_REDUCE intrinsic."""

  async def execute(self, target_executor: executor_base.Executor, arg):
    arg_type = self._type_signature.parameter
    seq = await _to_sequence(arg[0])
    accumulator, op = await asyncio.gather(*[
        _delegate(arg[idx], arg_type[idx], target_executor) for idx in [1, 2]
    ])
    element_type = seq.type_signature.element
    async for x in seq:
      el = await target_executor.create_value(x, element_type)
      arg = await target_executor.create_struct([accumulator, el])
      accumulator = await target_executor.create_call(op, arg)
    return accumulator


async def _to_sequence(sequence_val):
  """Returns an instance of `_Sequence` for given internal representation.

  Args:
    sequence_val: An internal sequence representation, either `_Sequence` or a
      value embedded in a target executor.

  Returns:
    An instance of `_Sequenc`.
  """
  if isinstance(sequence_val, _Sequence):
    return sequence_val
  py_typecheck.check_type(sequence_val, executor_value_base.ExecutorValue)
  return _SequenceFromPayload(await sequence_val.compute(),
                              sequence_val.type_signature)


async def _delegate(val, type_spec: computation_types.Type,
                    target_executor: executor_base.Executor):
  """Delegates value representation to target executor.

  Args:
    val: A value representation to delegate.
    type_spec: The TFF type.
    target_executor: The target executor to delegate.

  Returns:
    An instance of `executor_value_base.ExecutorValue` owned by the target
    executor.
  """
  py_typecheck.check_type(target_executor, executor_base.Executor)
  if val is None:
    return None
  if isinstance(val, executor_value_base.ExecutorValue):
    return val
  if isinstance(val, _Sequence):
    return await target_executor.create_value(await val.compute(), type_spec)
  if isinstance(val, structure.Struct):
    if len(val) != len(type_spec):
      raise ValueError('Found {} elements and {} types in a struct {}.'.format(
          len(val), len(type_spec), str(val)))
    elements = structure.iter_elements(val)
    element_types = structure.iter_elements(type_spec)
    names = []
    coros = []
    for (el_name, el), (el_type_name, el_type) in zip(elements, element_types):
      if el_name != el_type_name:
        raise ValueError(
            'Element name mismatch between value ({}) and type ({}).'.format(
                str(val), str(type_spec)))
      names.append(el_name)
      coros.append(_delegate(el, el_type, target_executor))
    flat_targets = await asyncio.gather(*coros)
    reassembled_struct = structure.Struct(list(zip(names, flat_targets)))
    return await target_executor.create_struct(reassembled_struct)
  return await target_executor.create_value(val, type_spec)


class SequenceExecutorValue(executor_value_base.ExecutorValue):
  """A representation of a value owned and managed by the `SequenceExecutor`."""

  _VALID_INTERNAL_REPRESENTATION_TYPES = (_Sequence, _SequenceOp,
                                          executor_value_base.ExecutorValue,
                                          structure.Struct)

  def __init__(self, value, type_spec):
    """Creates an instance of `SequenceExecutorValue` to represent `value`.

    The following kinds of representations are supported as the input:

    * An instance of `_Sequence` to represent sequences.

    * An instance of `_SequenceOp` for an instrinsic among one of the
      `SequenceExecutor._SUPPORTED_INTRINSIC_TO_SEQUENCE_OP` to represent the
      supported sequence intrinsics.

    * An instance of `ExecutorValue` (constructed and managed by a downstream
      executor).

    * An instance of `structure.Struct` with any of the above, possibly nested.

    Args:
      value: The internal representation of the value (see above).
      type_spec: The TFF type of the value.
    """
    py_typecheck.check_type(type_spec, computation_types.Type)
    py_typecheck.check_type(
        value, SequenceExecutorValue._VALID_INTERNAL_REPRESENTATION_TYPES)
    self._type_signature = type_spec
    self._value = value

  @property
  def internal_representation(self):
    return self._value

  @property
  def type_signature(self):
    return self._type_signature

  @tracing.trace
  async def compute(self):

    async def _comp(x):
      if isinstance(x, (_Sequence, executor_value_base.ExecutorValue)):
        return await x.compute()
      if isinstance(x, structure.Struct):
        return structure.pack_sequence_as(
            x, await asyncio.gather(*[_comp(y) for y in structure.flatten(x)]))
      raise NotImplementedError('Unable to compute a value of type {}.'.format(
          py_typecheck.type_string(type(x))))

    return await _comp(self._value)


class SequenceExecutor(executor_base.Executor):
  """The sequence executor is responsible for operations on sequences.

  NOTE: This executor is not fully implemented yet.

  This executor will understand TFF sequence types, sequence intrinsics, and
  potentially the `data` construct. It is intended for use as a building block
  for use in executor stacks that need to rely on sequence operators.

  This executor directly executes operations on sequences, as opposed to
  compiling or delegating them to another executor. Other executors may support
  operations on sequences, and they may handle them in a different, more
  efficient manner. This generic implementation is intended as a general-purpose
  building block that could be added at any level of the executor stack.

  Currently supported constructs:
  - tff.sequence_map
  - tff.sequence_reduce
  """

  def __init__(self, target_executor: executor_base.Executor):
    """Creates a new instance of this executor.

    Args:
      target_executor: Downstream executor that this executor delegates to.

    Raises:
      TypeError: if arguments are of the wrong types.
    """
    py_typecheck.check_type(target_executor, executor_base.Executor)
    self._target_executor = target_executor

  _SUPPORTED_INTRINSIC_TO_SEQUENCE_OP = {
      intrinsic_defs.SEQUENCE_MAP.uri: _SequenceMapOp,
      intrinsic_defs.SEQUENCE_REDUCE.uri: _SequenceReduceOp
  }

  @tracing.trace(span=True)
  async def create_value(self, value, type_spec=None):
    """Creates a value in this executor.

    The following kinds of `value` are supported as the input:

    * An instance of TFF computation proto containing one of the supported
      sequence intrinsics as its sole body.

    * An instance of eager TF dataset.

    * Anything that is supported by the target executor (as a pass-through).

    * A nested structure of any of the above.

    Args:
      value: The input for which to create a value.
      type_spec: An optional TFF type (required if `value` is not an instance of
        `typed_object.TypedObject`, otherwise it can be `None`).

    Returns:
      An instance of `SequenceExecutorValue` that represents the embedded value.
    """
    if type_spec is None:
      py_typecheck.check_type(value, typed_object.TypedObject)
      type_spec = value.type_signature
    else:
      type_spec = computation_types.to_type(type_spec)
    if isinstance(type_spec, computation_types.SequenceType):
      return SequenceExecutorValue(
          _SequenceFromPayload(value, type_spec), type_spec)
    if isinstance(value, pb.Computation):
      value_type = type_serialization.deserialize_type(value.type)
      value_type.check_equivalent_to(type_spec)
      which_computation = value.WhichOneof('computation')
      # NOTE: If not a supported type of intrinsic, we let it fall through and
      # be handled by embedding in the target executor (below).
      if which_computation == 'intrinsic':
        intrinsic_def = intrinsic_defs.uri_to_intrinsic_def(value.intrinsic.uri)
        if intrinsic_def is None:
          raise ValueError('Encountered an unrecognized intrinsic "{}".'.format(
              value.intrinsic.uri))
        op_type = SequenceExecutor._SUPPORTED_INTRINSIC_TO_SEQUENCE_OP.get(
            intrinsic_def.uri)
        if op_type is not None:
          type_analysis.check_concrete_instance_of(type_spec,
                                                   intrinsic_def.type_signature)
          op = op_type(type_spec)
          return SequenceExecutorValue(op, type_spec)
    if isinstance(type_spec, computation_types.StructType):
      if not isinstance(value, structure.Struct):
        value = structure.from_container(value)
      elements = structure.flatten(value)
      element_types = structure.flatten(type_spec)
      flat_embedded_vals = await asyncio.gather(*[
          self.create_value(el, el_type)
          for el, el_type in zip(elements, element_types)
      ])
      embedded_struct = structure.pack_sequence_as(value, flat_embedded_vals)
      return await self.create_struct(embedded_struct)
    target_value = await self._target_executor.create_value(value, type_spec)
    return SequenceExecutorValue(target_value, type_spec)

  @tracing.trace
  async def create_call(self, comp, arg=None):
    py_typecheck.check_type(comp, SequenceExecutorValue)
    py_typecheck.check_type(comp.type_signature, computation_types.FunctionType)
    fn = comp.internal_representation
    if isinstance(fn, executor_value_base.ExecutorValue):
      if arg is not None:
        arg = await _delegate(arg.internal_representation, arg.type_signature,
                              self._target_executor)
      target_result = await self._target_executor.create_call(fn, arg)
      return SequenceExecutorValue(target_result, target_result.type_signature)
    if isinstance(fn, _SequenceOp):
      py_typecheck.check_type(arg, SequenceExecutorValue)
      comp.type_signature.parameter.check_assignable_from(arg.type_signature)
      result = await fn.execute(self._target_executor,
                                arg.internal_representation)
      result_type = comp.type_signature.result
      return SequenceExecutorValue(result, result_type)
    raise NotImplementedError(
        'Unsupported functional representation of type {} (possibly indicating '
        'a mismatch between structures supported by create_cvalue() and '
        'create_call()).'.format(py_typecheck.type_string(type(fn))))

  @tracing.trace
  async def create_struct(self, elements):
    elements = structure.iter_elements(structure.from_container(elements))
    val_elements = []
    type_elements = []
    for k, v in elements:
      py_typecheck.check_type(v, SequenceExecutorValue)
      val_elements.append((k, v.internal_representation))
      type_elements.append((k, v.type_signature))
    return SequenceExecutorValue(
        structure.Struct(val_elements),
        computation_types.StructType(type_elements))

  @tracing.trace
  async def create_selection(self, source, index):
    py_typecheck.check_type(source, SequenceExecutorValue)
    py_typecheck.check_type(source.type_signature, computation_types.StructType)
    if isinstance(source.internal_representation,
                  executor_value_base.ExecutorValue):
      target_val = await self._target_executor.create_selection(
          source.internal_representation, index)
      return SequenceExecutorValue(target_val, target_val.type_signature)
    py_typecheck.check_type(source.internal_representation, structure.Struct)
    py_typecheck.check_type(index, int)
    return SequenceExecutorValue(source.internal_representation[index],
                                 source.type_signature[index])

  def close(self):
    self._target_executor.close()

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
"""An executor that caches and reuses values on repeated calls."""

import asyncio
import collections
import cachetools

import numpy as np
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_value_base


class HashableWrapper(collections.abc.Hashable):
  """A wrapper around non-hashable objects to be compared by identity."""

  def __init__(self, target):
    self._target = target

  def __hash__(self):
    return hash(id(self._target))

  def __eq__(self, other):
    return other is HashableWrapper and other._target is self._target  # pylint: disable=protected-access


def _get_hashable_key(value, type_spec):
  """Return a hashable key for value `value` of TFF type `type_spec`.

  Args:
    value: An argument to `create_value()`.
    type_spec: An optional type signature.

  Returns:
    A hashable key to use such that the same `value` always maps to the same
    key, and different ones map to different keys.

  Raises:
    TypeError: If there is no hashable key for this type of a value.
  """
  if type_spec.is_struct():
    if not isinstance(value, structure.Struct):
      try:
        value = structure.from_container(value)
      except Exception as e:
        raise TypeError(
            'Failed to convert value with type_spec {} to `Struct`'.format(
                repr(type_spec))) from e
    type_specs = structure.iter_elements(type_spec)
    r_elem = []
    for v, (field_name, field_type) in zip(value, type_specs):
      r_elem.append((field_name, _get_hashable_key(v, field_type)))
    return structure.Struct(r_elem)
  elif type_spec.is_federated():
    if type_spec.all_equal:
      return _get_hashable_key(value, type_spec.member)
    else:
      return tuple([_get_hashable_key(x, type_spec.member) for x in value])
  elif isinstance(value, pb.Computation):
    return value.SerializeToString(deterministic=True)
  elif isinstance(value, np.ndarray):
    return ('<dtype={},shape={}>'.format(value.dtype,
                                         value.shape), value.tobytes())
  elif (isinstance(value, collections.abc.Hashable) and
        not isinstance(value, (tf.Tensor, tf.Variable))):
    # TODO(b/139200385): Currently Tensor and Variable returns True for
    # `isinstance(value, collections.abc.Hashable)` even when it's not hashable.
    # Hence this workaround.
    return value
  else:
    return HashableWrapper(value)


class CachedValueIdentifier(collections.abc.Hashable):
  """An identifier for a cached value."""

  def __init__(self, identifier):
    py_typecheck.check_type(identifier, str)
    self._identifier = identifier

  def __hash__(self):
    return hash(self._identifier)

  def __eq__(self, other):
    # pylint: disable=protected-access
    return (isinstance(other, CachedValueIdentifier) and
            self._identifier == other._identifier)
    # pylint: enable=protected-access

  def __repr__(self):
    return 'CachedValueIdentifier({!r})'.format(self._identifier)

  def __str__(self):
    return self._identifier


class CachedValue(executor_value_base.ExecutorValue):
  """A value held by the caching executor."""

  def __init__(self, identifier, hashable_key, type_spec, target_future):
    """Creates a cached value.

    Args:
      identifier: An instance of `CachedValueIdentifier`.
      hashable_key: A hashable source value key, if any, or `None` of not
        applicable in this context, for use during cleanup.
      type_spec: The type signature of the target, an instance of `tff.Type`.
      target_future: An asyncio future that returns an instance of
        `executor_value_base.ExecutorValue` that represents a value embedded in
        the target executor.

    Raises:
      TypeError: If the arguments are of the wrong types.
    """
    py_typecheck.check_type(identifier, CachedValueIdentifier)
    py_typecheck.check_type(hashable_key, collections.abc.Hashable)
    py_typecheck.check_type(type_spec, computation_types.Type)
    if not asyncio.isfuture(target_future):
      raise TypeError('Expected an asyncio future, got {}'.format(
          py_typecheck.type_string(type(target_future))))
    self._identifier = identifier
    self._hashable_key = hashable_key
    self._type_spec = type_spec
    self._target_future = target_future
    self._computed_result = None

  @property
  def type_signature(self):
    return self._type_spec

  @property
  def identifier(self):
    return self._identifier

  @property
  def hashable_key(self):
    return self._hashable_key

  @property
  def target_future(self):
    return self._target_future

  async def compute(self):
    if self._computed_result is None:
      target_value = await self._target_future
      self._computed_result = await target_value.compute()
    return self._computed_result


_DEFAULT_CACHE_SIZE = 1000


class CachingExecutor(executor_base.Executor):
  """The caching executor only performs caching."""

  # TODO(b/134543154): Factor out default cache settings to supply elsewhere,
  # possibly as a part of executor stack configuration.

  # TODO(b/134543154): It might be desirable to still keep aorund things that
  # are currently in use (referenced) regardless of what's in the cache. This
  # can be added later on.

  def __init__(self, target_executor, cache=None):
    """Creates a new instance of this executor.

    Args:
      target_executor: An instance of `executor_base.Executor`.
      cache: The cache to use (must be an instance of `cachetools.Cache`). If
        unspecified, by default we construct a 1000-element LRU cache.
    """
    py_typecheck.check_type(target_executor, executor_base.Executor)
    if cache is not None:
      py_typecheck.check_type(cache, cachetools.Cache)
    else:
      cache = cachetools.LRUCache(_DEFAULT_CACHE_SIZE)
    self._target_executor = target_executor
    self._cache = cache
    self._num_values_created = 0

  def close(self):
    self._cache.clear()
    self._target_executor.close()

  def __del__(self):
    for k in list(self._cache):
      del self._cache[k]

  async def create_value(self, value, type_spec=None):
    type_spec = computation_types.to_type(type_spec)
    if isinstance(value, computation_impl.ComputationImpl):
      return await self.create_value(
          computation_impl.ComputationImpl.get_proto(value),
          type_utils.reconcile_value_with_type_spec(value, type_spec))
    py_typecheck.check_type(type_spec, computation_types.Type)
    hashable_key = _get_hashable_key(value, type_spec)
    try:
      identifier = self._cache.get(hashable_key)
    except TypeError as err:
      raise RuntimeError(
          'Failed to perform a hash table lookup with a value of Python '
          'type {} and TFF type {}, and payload {}: {}'.format(
              py_typecheck.type_string(type(value)), type_spec, value, err))
    if isinstance(identifier, CachedValueIdentifier):
      cached_value = self._cache.get(identifier)
      # If may be that the same payload appeared with a mismatching type spec,
      # which may be a legitimate use case if (as it happens) the payload alone
      # does not uniquely determine the type, so we simply opt not to reuse the
      # cache value and fallback on the regular behavior.
      if (cached_value is not None and type_spec is not None and
          not cached_value.type_signature.is_equivalent_to(type_spec)):
        identifier = None
    else:
      identifier = None
    if identifier is None:
      self._num_values_created = self._num_values_created + 1
      identifier = CachedValueIdentifier(str(self._num_values_created))
      self._cache[hashable_key] = identifier
      target_future = asyncio.ensure_future(
          self._target_executor.create_value(value, type_spec))
      cached_value = None
    if cached_value is None:
      cached_value = CachedValue(identifier, hashable_key, type_spec,
                                 target_future)
      self._cache[identifier] = cached_value
    try:
      await cached_value.target_future
    except Exception:
      # Invalidate the entire cache in the inner executor had an exception.
      # TODO(b/145514490): This is a bit heavy handed, there maybe caches where
      # only the current cache item needs to be invalidated; however this
      # currently only occurs when an inner RemoteExecutor has the backend go
      # down.
      self._cache = {}
      raise
    # No type check is necessary here; we have either checked
    # `is_equivalent_to` or just constructed `target_value`
    # explicitly with `type_spec`.
    return cached_value

  async def create_call(self, comp, arg=None):
    py_typecheck.check_type(comp, CachedValue)
    py_typecheck.check_type(comp.type_signature, computation_types.FunctionType)
    to_gather = [comp.target_future]
    if arg is not None:
      py_typecheck.check_type(arg, CachedValue)
      comp.type_signature.parameter.check_assignable_from(arg.type_signature)
      to_gather.append(arg.target_future)
      identifier_str = '{}({})'.format(comp.identifier, arg.identifier)
    else:
      identifier_str = '{}()'.format(comp.identifier)
    gathered = await asyncio.gather(*to_gather)
    type_spec = comp.type_signature.result
    identifier = CachedValueIdentifier(identifier_str)
    try:
      cached_value = self._cache[identifier]
    except KeyError:
      target_future = asyncio.ensure_future(
          self._target_executor.create_call(*gathered))
      cached_value = CachedValue(identifier, None, type_spec, target_future)
      self._cache[identifier] = cached_value
    try:
      target_value = await cached_value.target_future
    except Exception:
      # TODO(b/145514490): This is a bit heavy handed, there maybe caches where
      # only the current cache item needs to be invalidated; however this
      # currently only occurs when an inner RemoteExecutor has the backend go
      # down.
      self._cache = {}
      raise
    type_spec.check_assignable_from(target_value.type_signature)
    return cached_value

  async def create_struct(self, elements):
    if not isinstance(elements, structure.Struct):
      elements = structure.from_container(elements)
    element_strings = []
    element_kv_pairs = structure.to_elements(elements)
    to_gather = []
    type_elements = []
    for k, v in element_kv_pairs:
      py_typecheck.check_type(v, CachedValue)
      to_gather.append(v.target_future)
      if k is not None:
        py_typecheck.check_type(k, str)
        element_strings.append('{}={}'.format(k, v.identifier))
        type_elements.append((k, v.type_signature))
      else:
        element_strings.append(str(v.identifier))
        type_elements.append(v.type_signature)
    type_spec = computation_types.StructType(type_elements)
    gathered = await asyncio.gather(*to_gather)
    identifier = CachedValueIdentifier('<{}>'.format(','.join(element_strings)))
    try:
      cached_value = self._cache[identifier]
    except KeyError:
      target_future = asyncio.ensure_future(
          self._target_executor.create_struct(
              structure.Struct(
                  (k, v) for (k, _), v in zip(element_kv_pairs, gathered))))
      cached_value = CachedValue(identifier, None, type_spec, target_future)
      self._cache[identifier] = cached_value
    try:
      target_value = await cached_value.target_future
    except Exception:
      # TODO(b/145514490): This is a bit heavy handed, there maybe caches where
      # only the current cache item needs to be invalidated; however this
      # currently only occurs when an inner RemoteExecutor has the backend go
      # down.
      self._cache = {}
      raise
    type_spec.check_assignable_from(target_value.type_signature)
    return cached_value

  async def create_selection(self, source, index=None, name=None):
    py_typecheck.check_type(source, CachedValue)
    py_typecheck.check_type(source.type_signature, computation_types.StructType)
    source_val = await source.target_future
    if index is not None:
      py_typecheck.check_none(name)
      identifier_str = '{}[{}]'.format(source.identifier, index)
      type_spec = source.type_signature[index]
    else:
      py_typecheck.check_not_none(name)
      identifier_str = '{}.{}'.format(source.identifier, name)
      type_spec = getattr(source.type_signature, name)
    identifier = CachedValueIdentifier(identifier_str)
    try:
      cached_value = self._cache[identifier]
    except KeyError:
      target_future = asyncio.ensure_future(
          self._target_executor.create_selection(
              source_val, index=index, name=name))
      cached_value = CachedValue(identifier, None, type_spec, target_future)
      self._cache[identifier] = cached_value
    try:
      target_value = await cached_value.target_future
    except Exception:
      # TODO(b/145514490): This is a bit heavy handed, there maybe caches where
      # only the current cache item needs to be invalidated; however this
      # currently only occurs when an inner RemoteExecutor has the backend go
      # down.
      self._cache = {}
      raise
    type_spec.check_assignable_from(target_value.type_signature)
    return cached_value

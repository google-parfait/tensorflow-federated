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
"""An executor that caches and reuses values on repeated calls."""

import collections
import weakref

import numpy as np

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import executor_value_base
from tensorflow_federated.python.core.impl import type_utils


class HashableWrapper(collections.Hashable):
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
  if isinstance(type_spec, computation_types.NamedTupleType):
    if isinstance(value, anonymous_tuple.AnonymousTuple):
      v_elem = anonymous_tuple.to_elements(value)
      t_elem = anonymous_tuple.to_elements(type_spec)
      r_elem = []
      for (_, vv), (tk, tv) in zip(v_elem, t_elem):
        r_elem.append((tk, _get_hashable_key(vv, tv)))
      return anonymous_tuple.AnonymousTuple(r_elem)
    else:
      return _get_hashable_key(anonymous_tuple.from_container(value), type_spec)
  elif isinstance(type_spec, computation_types.FederatedType):
    if type_spec.all_equal:
      return _get_hashable_key(value, type_spec.member)
    else:
      return tuple([_get_hashable_key(x, type_spec.member) for x in value])
  elif isinstance(value, pb.Computation):
    return str(value)
  elif isinstance(value, np.ndarray):
    # TODO(b/138437499): Find something more efficient.
    return '<dtype={},shape={},items={}>'.format(value.dtype, value.shape,
                                                 value.flatten())
  elif isinstance(value, collections.Hashable):
    return value
  else:
    return HashableWrapper(value)


class CachingExecutor(executor_base.Executor):
  """The caching executor only performs caching."""

  # TODO(b/138437499): It might be desirable for this caching executor to keep
  # things in the cache after they are no longer referenced (the current policy
  # is to only reuse a value that is still actively being referenced someplace
  # else, which may or may not be adequate depending on the situation).

  def __init__(self, target_executor):
    """Creates a new instance of this executor.

    Args:
      target_executor: An instance of `executor_base.Executor`.
    """
    py_typecheck.check_type(target_executor, executor_base.Executor)
    self._target_executor = target_executor
    self._num_values_created = 0
    self._hashable_key_to_identifier = {}
    self._identifier_to_cached_value = {}

  def __del__(self):
    for _, v in self._identifier_to_cached_value.items():
      target_value = v.target_value
      if target_value is not None:
        del target_value
    self._hashable_key_to_identifier.clear()
    self._identifier_to_cached_value.clear()

  async def create_value(self, value, type_spec=None):
    type_spec = computation_types.to_type(type_spec)
    if isinstance(value, computation_impl.ComputationImpl):
      return await self.create_value(
          computation_impl.ComputationImpl.get_proto(value),
          type_utils.reconcile_value_with_type_spec(value, type_spec))
    hashable_key = _get_hashable_key(value, type_spec)
    try:
      identifier = self._hashable_key_to_identifier.get(hashable_key)
    except TypeError as err:
      raise RuntimeError(
          'Failed to perform a has table lookup with a value of Python '
          'type {} and TFF type {}, and payload {}: {}'.format(
              py_typecheck.type_string(type(value)), type_spec, value, err))
    if identifier is not None:
      py_typecheck.check_type(identifier, str)
      cached_value = self._identifier_to_cached_value.get(identifier)
      # If may be that the same payload appeared with a mismatching type spec,
      # which may be a legitimate use case if (as it happens) the payload alone
      # does not uniquely determine the type, so we simply opt not to reuse the
      # cache value and fallback on the regular behavior.
      if type_spec is not None and not type_utils.are_equivalent_types(
          cached_value.type_signature, type_spec):
        identifier = None
    if identifier is None:
      target_value = await self._target_executor.create_value(value, type_spec)
      self._num_values_created = self._num_values_created + 1
      identifier = str(self._num_values_created)
      self._hashable_key_to_identifier[hashable_key] = identifier
      cached_value = None
    if cached_value is None:
      cached_value = CachedValue(self, identifier, hashable_key, target_value)
      self._identifier_to_cached_value[identifier] = cached_value
    return cached_value

  def delete_value(self, value):
    py_typecheck.check_type(value, CachedValue)
    try:
      del self._identifier_to_cached_value[value.identifier]
    except KeyError:
      pass
    if value.hashable_key is not None:
      try:
        del self._hashable_key_to_identifier[value.hashable_key]
      except KeyError:
        pass

  async def create_call(self, comp, arg=None):
    py_typecheck.check_type(comp, CachedValue)
    if arg is not None:
      py_typecheck.check_type(arg, CachedValue)
      identifier = '{}({})'.format(comp.identifier, arg.identifier)
    else:
      identifier = '{}()'.format(comp.identifier)
    cached_value = self._identifier_to_cached_value.get(identifier)
    if cached_value is None:
      target_value = await self._target_executor.create_call(
          comp.target_value, arg.target_value if arg is not None else None)
      cached_value = CachedValue(self, identifier, None, target_value)
      self._identifier_to_cached_value[identifier] = cached_value
    return cached_value

  async def create_tuple(self, elements):
    if not isinstance(elements, anonymous_tuple.AnonymousTuple):
      elements = anonymous_tuple.from_container(elements)
    element_strings = []
    for k, v in anonymous_tuple.to_elements(elements):
      py_typecheck.check_type(v, CachedValue)
      if k is not None:
        py_typecheck.check_type(k, str)
        element_strings.append('{}={}'.format(k, v.identifier))
      else:
        element_strings.append(v.identifier)
    identifier = '<{}>'.format(','.join(element_strings))
    cached_value = self._identifier_to_cached_value.get(identifier)
    if cached_value is None:
      target_value = await self._target_executor.create_tuple(
          anonymous_tuple.map_structure(lambda x: x.target_value, elements))
      cached_value = CachedValue(self, identifier, None, target_value)
      self._identifier_to_cached_value[identifier] = cached_value
    return cached_value

  async def create_selection(self, source, index=None, name=None):
    py_typecheck.check_type(source, CachedValue)
    if index is not None:
      py_typecheck.check_none(name)
      identifier = '{}[{}]'.format(source.identifier, index)
    else:
      py_typecheck.check_not_none(name)
      identifier = '{}.{}'.format(source.identifier, name)
    cached_value = self._identifier_to_cached_value.get(identifier)
    if cached_value is None:
      target_value = await self._target_executor.create_selection(
          source.target_value, index=index, name=name)
      cached_value = CachedValue(self, identifier, None, target_value)
      self._identifier_to_cached_value[identifier] = cached_value
    return cached_value


class CachedValue(executor_value_base.ExecutorValue):
  """A value held by the caching executor."""

  def __init__(self, owner, identifier, hashable_key, target_value):
    """Creates a cached value.

    Args:
      owner: An instance of `CachingExecutor`.
      identifier: A string that uniquely identifies this value.
      hashable_key: A hashable source value key, if any, or `None` of not
        applicable in this context, for use during cleanup.
      target_value: An instance of `executor_value_base.ExecutorValue` that
        represents a value embedded in the target executor.
    """
    py_typecheck.check_type(owner, CachingExecutor)
    py_typecheck.check_type(identifier, str)
    py_typecheck.check_type(hashable_key, collections.Hashable)
    py_typecheck.check_type(target_value, executor_value_base.ExecutorValue)
    self._owner = weakref.ref(owner)
    self._identifier = identifier
    self._hashable_key = hashable_key
    self._target_value = target_value
    self._computed_result = None

  def __del__(self):
    del self._computed_result
    del self._target_value
    owner = self._owner()
    if owner is not None:
      owner.delete_value(self)

  @property
  def type_signature(self):
    return self._target_value.type_signature

  @property
  def identifier(self):
    return self._identifier

  @property
  def hashable_key(self):
    return self._hashable_key

  @property
  def target_value(self):
    return self._target_value

  async def compute(self):
    if self._computed_result is None:
      self._computed_result = await self._target_value.compute()
    return self._computed_result

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
"""Expands ClientData by performing transformations."""

import bisect
import re
from typing import Any, Callable, List, Optional

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.simulation.datasets import client_data

CLIENT_ID_REGEX = re.compile(r'^(.*)_(\d+)$')


class TransformingClientData(client_data.ClientData):
  """Transforms client data, potentially expanding by adding pseudo-clients.

  Each client of the raw_client_data is "expanded" into some number of
  pseudo-clients. Each client ID is a string consisting of the original client
  ID plus a concatenated integer index. For example, the raw client id
  "client_a" might be expanded into pseudo-client ids "client_a_0", "client_a_1"
  and "client_a_2". A function fn(x) maps datapoint x to a new datapoint,
  where the constructor of fn is parameterized by the (raw) client_id and index
  i. For example if x is an image, then make_transform_fn("client_a", 0)(x)
  might be the identity, while make_transform_fn("client_a", 1)(x) could be a
  random rotation of the image with the angle determined by a hash of "client_a"
  and "1". Typically by convention the index 0 corresponds to the identity
  function if the identity is supported.
  """

  def __init__(self,
               raw_client_data: client_data.ClientData,
               make_transform_fn: Callable[[str, int], Callable[[Any], Any]],
               num_transformed_clients: Optional[int] = None):
    """Initializes the TransformingClientData.

    Args:
      raw_client_data: A ClientData to expand.
      make_transform_fn: A function that returns a callable that maps datapoint
        x to a new datapoint x'. make_transform_fn will be called as
        make_transform_fn(raw_client_id, i) where i is an integer index, and
        should return a function fn(x)->x. For example if x is an image, then
        make_transform_fn("client_a", 0)(x) might be the identity, while
        make_transform_fn("client_a", 1)(x) could be a random rotation of the
        image with the angle determined by a hash of "client_a" and "1". If
        transform_fn_cons returns `None`, no transformation is performed.
        Typically by convention the index 0 corresponds to the identity function
        if the identity is supported.
      num_transformed_clients: The total number of transformed clients to
        produce. If `None`, only the original clients will be transformed. If it
        is an integer multiple k of the number of real clients, there will be
        exactly k pseudo-clients per real client, with indices 0...k-1. Any
        remainder g will be generated from the first g real clients and will be
        given index k.
    """
    py_typecheck.check_type(raw_client_data, client_data.ClientData)
    py_typecheck.check_callable(make_transform_fn)

    raw_client_ids = raw_client_data.client_ids
    if not raw_client_ids:
      raise ValueError('`raw_client_data` must be non-empty.')

    if num_transformed_clients is None:
      num_transformed_clients = len(raw_client_ids)
    else:
      py_typecheck.check_type(num_transformed_clients, int)
      if num_transformed_clients <= 0:
        raise ValueError('`num_transformed_clients` must be positive.')

    self._raw_client_data = raw_client_data
    self._make_transform_fn = make_transform_fn

    self._has_pseudo_clients = num_transformed_clients > len(raw_client_ids)

    if self._has_pseudo_clients:
      num_digits = len(str(num_transformed_clients - 1))
      format_str = '{}_{:0' + str(num_digits) + '}'

      k = num_transformed_clients // len(raw_client_ids)
      self._client_ids = []
      for raw_client_id in raw_client_ids:
        for i in range(k):
          self._client_ids.append(format_str.format(raw_client_id, i))
      num_extra_client_ids = num_transformed_clients - k * len(raw_client_ids)
      for c in range(num_extra_client_ids):
        self._client_ids.append(format_str.format(raw_client_ids[c], k))
    else:
      self._client_ids = raw_client_ids

    # Already sorted if raw_client_data.client_ids are, but just to be sure...
    self._client_ids = sorted(self._client_ids)

  @property
  def client_ids(self) -> List[str]:
    return self._client_ids

  def split_client_id(self, client_id):
    """Splits pseudo-client id into raw client id and index components.

    Args:
      client_id: The pseudo-client id.

    Returns:
      A tuple (raw_client_id, index) where raw_client_id is the string of the
      raw client_id, and index is the integer index of the pseudo-client.
    """
    if not self._has_pseudo_clients:
      return client_id, 0

    py_typecheck.check_type(client_id, str)
    match = CLIENT_ID_REGEX.search(client_id)
    if not match:
      raise ValueError('client_id must be a valid string from client_ids.')
    raw_client_id = match.group(1)
    index = int(match.group(2))
    return raw_client_id, index

  def create_tf_dataset_for_client(self, client_id: str) -> tf.data.Dataset:
    py_typecheck.check_type(client_id, str)
    i = bisect.bisect_left(self._client_ids, client_id)
    if i == len(self._client_ids) or self._client_ids[i] != client_id:
      raise ValueError('client_id must be a valid string from client_ids.')

    raw_client_id, index = self.split_client_id(client_id)
    raw_dataset = self._raw_client_data.create_tf_dataset_for_client(
        raw_client_id)

    transform_fn = self._make_transform_fn(raw_client_id, index)
    if not transform_fn:
      return raw_dataset
    else:
      py_typecheck.check_callable(transform_fn)
      return raw_dataset.map(transform_fn, tf.data.experimental.AUTOTUNE)

  @property
  def element_type_structure(self):
    return self._raw_client_data.element_type_structure

  @property
  def dataset_computation(self):
    raise NotImplementedError('TranscormingClientData contains non-TensorFlow '
                              'logic and is currently incompatible with '
                              'dataset_computation.')

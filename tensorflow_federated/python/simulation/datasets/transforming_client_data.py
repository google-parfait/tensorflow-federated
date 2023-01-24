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

from collections.abc import Callable
from typing import Any, Optional

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.simulation.datasets import client_data


class TransformingClientData(client_data.ClientData):
  """Transforms client data, potentially expanding by adding pseudo-clients.

  Each client of the base_client_data is "expanded" into some number of
  pseudo-clients. A serializable function fn(x) maps datapoint x to a new
  datapoint, where the constructor of fn is parameterized by the expanded
  client_id. For example if the client_id "client_A" has two expansions,
  "client_A-0" and "client_A-1" then make_transform_fn("client_A-0")(x) might be
  the identity, while make_transform_fn("client_A-1")(x) could be a random
  rotation of the image with the angle determined by a hash of the string
  "client_A-1".
  """

  def __init__(
      self,
      base_client_data: client_data.ClientData,
      make_transform_fn: Callable[[str], Callable[[Any], Any]],
      expand_client_id: Optional[Callable[[str], list[str]]] = None,
      reduce_client_id: Optional[Callable[[str], str]] = None,
  ):
    """Initializes the TransformingClientData.

    Args:
      base_client_data: A ClientData to expand.
      make_transform_fn: A function to be called as
        `make_transform_fn(client_id)`, where `client_id` is the expanded client
        id, which should return a function `transform_fn` that maps a datapoint
        x whose element type structure correspondes to `base_client_data` to a
        new datapoint x'. It must be traceable as a `tf.function`.
      expand_client_id: An optional function that maps a client id of
        `base_client_data` to a list of expanded client ids. If None, the
        transformed data will have the same size and ids as the original.
      reduce_client_id: An function that maps an expanded client id back to the
        raw client id. Must be traceable as a `tf.function`. Must be specified
        if and only if `expand_client_id` is.
    """
    py_typecheck.check_type(base_client_data, client_data.ClientData)
    py_typecheck.check_callable(make_transform_fn)

    raw_client_ids = base_client_data.client_ids
    if not raw_client_ids:
      raise ValueError('`base_client_data` must be non-empty.')

    self._base_client_data = base_client_data
    self._make_transform_fn = make_transform_fn

    if (expand_client_id is None) != (reduce_client_id is None):
      raise ValueError(
          'Must specify both or neither of `expand_client_id` and '
          '`reduce_client_id`.'
      )

    if expand_client_id is None:
      self._client_ids = raw_client_ids
      # pylint: disable=unnecessary-lambda
      self._reduce_client_id = lambda s: tf.convert_to_tensor(s)
      # pylint: enable=unnecessary-lambda
    else:
      self._client_ids = []
      for client_id in raw_client_ids:
        for expanded_client_id in expand_client_id(client_id):
          self._client_ids.append(expanded_client_id)
      self._reduce_client_id = reduce_client_id

    self._client_ids = sorted(self._client_ids)

    example_dataset = self._create_dataset(next(iter(self._client_ids)))
    self._element_type_structure = example_dataset.element_spec

  @property
  def client_ids(self) -> list[str]:
    return self._client_ids

  def _create_dataset(self, client_id: str) -> tf.data.Dataset:
    orig_client_id = self._reduce_client_id(client_id)
    orig_dataset = self._base_client_data.serializable_dataset_fn(
        orig_client_id
    )
    transform = self._make_transform_fn(client_id)
    return orig_dataset.map(transform, tf.data.experimental.AUTOTUNE)

  @property
  def serializable_dataset_fn(self):
    return self._create_dataset

  @property
  def element_type_structure(self):
    return self._element_type_structure

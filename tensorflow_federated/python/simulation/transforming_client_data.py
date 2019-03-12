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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect
import re

from six.moves import range

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.simulation import client_data


class TransformingClientData(client_data.ClientData):
  """Expands client data by performing transformations.

  Each client of the raw_client_data is "expanded" into some number of
  pseudo-clients. Each client ID is a tuple containing the original client ID
  plus an integer index. A function f(x, i) maps datapoints x with index i to
  new datapoint. For example if x is an image, and i has values 0 or 1, f(x, 0)
  might be the identity, while f(x, 1) could be the reflection of the image.
  """

  def __init__(self, raw_client_data, transform_fn, num_transformed_clients):
    """Initializes the TransformingClientData.

    Args:
      raw_client_data: A ClientData to expand.
      transform_fn: A function f(x, i) parameterized by i, mapping datapoint x
        to a new datapoint. x is a datapoint from the raw_client_data, while i
        is an integer index in the range 0...k (see 'num_transformed_clients'
        for definition of k). Typically by convention the index 0 corresponds to
        the identity function if the identity is supported.
      num_transformed_clients: The total number of transformed clients to
        produce. If it is an integer multiple k of the number of real clients,
        there will be exactly k pseudo-clients per real client, with indices
        0...k-1. Any remainder g will be generated from the first g real clients
        and will be given index k.
    """
    py_typecheck.check_type(raw_client_data, client_data.ClientData)
    py_typecheck.check_callable(transform_fn)
    py_typecheck.check_type(num_transformed_clients, int)

    if num_transformed_clients <= 0:
      raise ValueError('num_transformed_clients must be positive and finite.')
    self._raw_client_data = raw_client_data
    self._transform_fn = transform_fn

    num_digits = len(str(num_transformed_clients))
    format_str = '{}_{:0' + str(num_digits) + '}'

    raw_client_ids = raw_client_data.client_ids
    k = num_transformed_clients // len(raw_client_ids)
    self._client_ids = []
    for raw_client_id in raw_client_ids:
      for i in range(k):
        self._client_ids.append(format_str.format(raw_client_id, i))
    num_extra_client_ids = num_transformed_clients - k * len(raw_client_ids)
    for c in range(num_extra_client_ids):
      self._client_ids.append(format_str.format(raw_client_ids[c], k))

    # Already sorted if raw_client_data.client_ids are, but just to be sure...
    self._client_ids = sorted(self._client_ids)

  @property
  def client_ids(self):
    return self._client_ids

  def create_tf_dataset_for_client(self, client_id):
    py_typecheck.check_type(client_id, str)
    i = bisect.bisect_left(self._client_ids, client_id)
    if i == len(self._client_ids) or self._client_ids[i] != client_id:
      raise ValueError('client_id must be a valid string from client_ids.')
    pattern = r'^(.*)_(\d*)$'
    match = re.search(pattern, client_id)
    if not match:
      # This should be impossible if client_id is in self._client_ids.
      raise ValueError('client_id must be a valid string from client_ids.')
    raw_client_id = match.group(1)
    expansion_id = int(match.group(2))
    raw_dataset = self._raw_client_data.create_tf_dataset_for_client(
        raw_client_id)

    def _transform_fn_wrapper(example):
      return self._transform_fn(example, expansion_id)

    return raw_dataset.map(_transform_fn_wrapper)

  @property
  def output_types(self):
    return self._raw_client_data.output_types

  @property
  def output_shapes(self):
    return self._raw_client_data.output_shapes

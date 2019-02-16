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

import math
import re

import numpy as np
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

  def __init__(self, raw_client_data, transform, expansion_factor):
    """Initializes the TransformingClientData.

    Args:
      raw_client_data: A ClientData to expand.
      transform: A function f(x, i) mapping datapoint x to a new datapoint
        parameterized by i.
      expansion_factor: The (expected) number of transformed clients per raw
        client. If not an integer, each client is mapped to at least
        int(expansion_factor) new clients, and some fraction of clients are
        mapped to one more.
    """
    py_typecheck.check_type(raw_client_data, client_data.ClientData)
    py_typecheck.check_callable(transform)

    if expansion_factor <= 0 or math.isinf(expansion_factor):
      raise ValueError('expansion_factor must be positive and finite.')
    self._raw_client_data = raw_client_data
    self._transform = transform

    raw_client_ids = raw_client_data.client_ids
    num_entire_client_ids = int(expansion_factor)
    self._client_ids = []
    for raw_client_id in raw_client_ids:
      for i in range(num_entire_client_ids):
        self._client_ids.append('{}_{}'.format(raw_client_id, i))
    num_extra_client_ids = int(
        len(raw_client_ids) * (expansion_factor - num_entire_client_ids))
    if num_extra_client_ids > 0:
      extra_client_ids = np.random.choice(
          raw_client_ids, num_extra_client_ids, replace=False)
      for raw_client_id in extra_client_ids:
        self._client_ids.append('{}_{}'.format(raw_client_id,
                                               num_entire_client_ids))
    self._client_ids = sorted(self._client_ids)

  @property
  def client_ids(self):
    return self._client_ids

  def create_tf_dataset_for_client(self, client_id):
    py_typecheck.check_type(client_id, str)
    pattern = r'^(.*)_(\d*)$'
    match = re.search(pattern, client_id)
    if not match:
      raise ValueError('client_id must be a valid string from client_ids.')
    raw_client_id = match.group(1)
    expansion_id = int(match.group(2))
    raw_dataset = self._raw_client_data.create_tf_dataset_for_client(
        raw_client_id)
    return raw_dataset.map(lambda x: self._transform(x, expansion_id))

  @property
  def output_types(self):
    return self._raw_client_data.output_types

  @property
  def output_shapes(self):
    return self._raw_client_data.output_shapes

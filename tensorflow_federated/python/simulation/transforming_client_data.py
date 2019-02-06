# Copyright 2018, The TensorFlow Federated Authors.
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

import numpy as np
from six.moves import zip

from tensorflow_federated.python.simulation import client_data


class TransformingClientData(client_data.ClientData):
  """Expands client data by performing transformations.

  Each client of the raw_client_data is "expanded" into some number of new
  pseudo-clients.
  """

  def __init__(self, raw_client_data, transform, expansion_factor):
    self._raw_client_data = raw_client_data
    self._transform = transform

    raw_client_ids = raw_client_data.client_ids
    num_entire_client_ids = int(expansion_factor)
    self._client_ids = []
    for i in range(num_entire_client_ids):
      self._client_ids.extend(zip(raw_client_ids, [i] * len(raw_client_ids)))
    num_extra_client_ids = int(
        len(raw_client_ids) * (expansion_factor - num_entire_client_ids))
    extra_client_ids = np.random.choice(
        raw_client_ids, num_extra_client_ids, replace=False)
    self._client_ids.extend(
        zip(extra_client_ids, [num_entire_client_ids] * num_extra_client_ids))

  @property
  def client_ids(self):
    return self._client_ids

  def create_tf_dataset_for_client(self, client_id):
    raw_client_id, expansion_id = client_id
    raw_dataset = self._raw_client_data.create_tf_dataset_for_client(
        raw_client_id)
    return raw_dataset.map(lambda x: self._transform(x, expansion_id))

  def output_types(self):
    return self._raw_client_data.output_types

  def output_shapes(self):
    return self._raw_client_data.output_shapes

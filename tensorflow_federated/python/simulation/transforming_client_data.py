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

from typing import Any, Callable, Optional
import warnings

from tensorflow_federated.python.simulation.datasets import client_data
from tensorflow_federated.python.simulation.datasets import transforming_client_data

# TODO(b/182305417): Delete this once the full deprecation period has passed.


class TransformingClientData(transforming_client_data.TransformingClientData):
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

  WARNING: this class is deprecated and is slated for removal in April 2021.
  Please use `tff.simulation.datasets.TransformingClientData` instead.
  """

  def __init__(self,
               raw_client_data: client_data.ClientData,
               make_transform_fn: Callable[[str, int], Callable[[Any], Any]],
               num_transformed_clients: Optional[int] = None):
    """Initializes the TransformingClientData.

    WARNING: this class is deprecated and is slated for removal in April 2021.
    Please use `tff.simulation.datasets.TransformingClientData` instead.

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
    warnings.warn(
        'tff.simulation.TransformingClientData is deprecated and slated for '
        'removal in April 2021. Please use '
        'tff.simulation.datasets.TransformingClientData instead.',
        DeprecationWarning)
    transforming_client_data.TransformingClientData.__init__(
        self,
        raw_client_data,
        make_transform_fn,
        num_transformed_clients=num_transformed_clients)

# Copyright 2020, The TensorFlow Federated Authors.
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
"""Simple temperature sensor example in TFF."""

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


@tff.tf_computation(tff.SequenceType(tf.float32), tf.float32)
def count_over(ds, t):
  return ds.reduce(
      np.float32(0), lambda n, x: n + tf.cast(tf.greater(x, t), tf.float32))


@tff.tf_computation(tff.SequenceType(tf.float32))
def count_total(ds):
  return ds.reduce(np.float32(0.0), lambda n, _: n + 1.0)


@tff.federated_computation(
    tff.FederatedType(tff.SequenceType(tf.float32), tff.CLIENTS),
    tff.FederatedType(tf.float32, tff.SERVER))
def mean_over_threshold(temperatures, threshold):
  client_data = tff.federated_broadcast(threshold)
  client_data = tff.federated_zip([temperatures, client_data])
  result_map = tff.federated_map(count_over, client_data)
  count_map = tff.federated_map(count_total, temperatures)
  return tff.federated_mean(result_map, count_map)

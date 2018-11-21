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
"""Tests for types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

import unittest

from tensorflow_federated.python.core.api.computations import federated_computation
from tensorflow_federated.python.core.api.computations import tf_computation
from tensorflow_federated.python.core.api.intrinsics import federated_broadcast
from tensorflow_federated.python.core.api.intrinsics import federated_map
from tensorflow_federated.python.core.api.intrinsics import federated_sum
from tensorflow_federated.python.core.api.placements import SERVER
from tensorflow_federated.python.core.api.types import FederatedType


class IntrinsicsTest(unittest.TestCase):

  def test_simple(self):
    @federated_computation(FederatedType(tf.float32, SERVER, True))
    def foo(x):
      return federated_sum(
          federated_map(
              federated_broadcast(x),
              tf_computation(lambda x: tf.to_int32(x > 0.5), tf.float32)))
    self.assertEqual(
        str(foo.type_signature), '(float32@SERVER -> int32@SERVER)')


if __name__ == '__main__':
  unittest.main()

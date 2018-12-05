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
"""Tests for sanity."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core import api as fc


class SanityTest(absltest.TestCase):

  def test_core_api(self):
    @fc.federated_computation(fc.FederatedType(tf.bool, fc.CLIENTS))
    def foo(x):
      return fc.federated_sum(
          fc.federated_map(x, fc.tf_computation(tf.to_int32, tf.bool)))
    self.assertEqual(
        str(foo.type_signature), '({bool}@CLIENTS -> int32@SERVER)')


if __name__ == '__main__':
  absltest.main()

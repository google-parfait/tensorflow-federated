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
import tensorflow as tf

import unittest

from tensorflow_federated.python.core import api as fc


class SanityTest(unittest.TestCase):

  def test_sanity(self):
    @fc.federated_computation(fc.FederatedType(tf.int32, fc.CLIENTS))
    def foo(x):
      return x
    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> {int32}@CLIENTS)')

  def test_core_api(self):
    self.assertEqual(str(fc.to_type(tf.int32)), 'int32')


if __name__ == '__main__':
  unittest.main()

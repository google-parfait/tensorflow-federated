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
"""End-to-end example testing TensorFlow Federated against the MNIST model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


import numpy as np
import tensorflow as tf

from tensorflow_federated import python as tff

from tensorflow_federated.python.examples.mnist import mnist


class MnistTest(tf.test.TestCase):

  def test_something(self):
    it_process = tff.learning.build_federated_averaging_process(mnist.model_fn)
    self.assertIsInstance(it_process, tff.utils.IterativeProcess)
    federated_data_type = it_process.next.type_signature.parameter[1]
    self.assertEqual(
        str(federated_data_type), '{<x=float32[?,784],y=int64[?,1]>*}@CLIENTS')

  def test_simple_training(self):
    it_process = tff.learning.build_federated_averaging_process(mnist.model_fn)
    server_state = it_process.initialize()
    Batch = collections.namedtuple('Batch', ['x', 'y'])  # pylint: disable=invalid-name

    def deterministic_batch():
      return Batch(
          x=np.ones([1, 784], dtype=np.float32),
          y=np.ones([1, 1], dtype=np.int64))

    batch = tff.tf_computation(deterministic_batch)()
    federated_data = [[batch]]

    next_state, orig_loss = it_process.next(server_state, federated_data)
    loss_list = []
    for _ in range(2):
      next_state, loss = it_process.next(next_state, federated_data)
      loss_list.append(loss)
    self.assertLess(np.mean(loss_list), orig_loss)


if __name__ == '__main__':
  tf.enable_resource_variables()
  tf.test.main()

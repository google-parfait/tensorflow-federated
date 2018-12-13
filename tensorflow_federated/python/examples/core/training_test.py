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
"""Tests for training.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_federated import python as tff

from tensorflow_federated.python.examples.core import training


class TrainingTest(tf.test.TestCase):

  def test_forward_pass(self):
    # Just testing here for now that we can compute the loss, that all types and
    # shapes match, etc.

    # Since execution is not yet supported, using `tf_computation` temporarily
    # to allow `forward_pass` to get stamped, and manually driving Session.run
    # to execute it.
    # TODO(b/113116813): Use the interfaces for executing computations as soon
    # as they're ready rather than manually driving TensorFlow graphs in tests,
    # which should help to shorten and simplify this code.
    @tff.tf_computation
    def _():
      # When creating the batch, we have to specify a concrete shape, since by
      # default 'BATCH_TYPE' leaves batch size undefined.
      batch = tff.utils.get_variables(
          'batch', [('X', (tf.int32, [5])), ('Y', (tf.int32, [5]))],
          initializer=tf.zeros_initializer())

      model = tff.utils.get_variables(
          'model', training.MODEL_TYPE, initializer=tf.zeros_initializer())

      loss = training.forward_pass(batch, model).loss

      # TODO(b/113116813): Replace this temporary workaround with a proper call
      # that gets plumbed through the execution API, once it materializes.
      # For now, just testing here that the graph has been stitched correctly,
      # and that something gets computed at all.
      with tf.Session(graph=tf.get_default_graph()) as sess:
        sess.run(tf.global_variables_initializer())
        self.assertIsInstance(sess.run(loss), np.float32)

      return loss


if __name__ == '__main__':
  tf.test.main()

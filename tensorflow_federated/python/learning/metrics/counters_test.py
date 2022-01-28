# Copyright 2022, The TensorFlow Federated Authors.
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

import tensorflow as tf

from tensorflow_federated.python.learning.metrics import counters


class NumExamplesCounterTest(tf.test.TestCase):

  def test_construct(self):
    m = counters.NumExamplesCounter()
    self.assertEqual(m.name, 'num_examples')
    self.assertTrue(m.stateful)
    self.assertEqual(m.dtype, tf.int64)
    self.assertLen(m.variables, 1)
    self.assertEqual(m.total, 0)
    m = counters.NumExamplesCounter('num_examples2')
    self.assertEqual(m.name, 'num_examples2')

  def test_update_without_sample_weight(self):
    m = counters.NumExamplesCounter()
    self.assertEqual(m(tf.zeros([10, 1]), tf.zeros([10])), 10)
    self.assertEqual(m.total, 10)
    self.assertEqual(m.update_state(tf.zeros([5, 1]), tf.zeros([5])), 15)
    self.assertEqual(m.total, 15)

  def test_update_with_sample_weight(self):
    m = counters.NumExamplesCounter()
    self.assertEqual(
        m(tf.zeros([10, 1]), tf.zeros([10]), sample_weight=0.5), 10)
    self.assertEqual(m.total, 10)
    self.assertEqual(
        m.update_state(tf.zeros([5, 1]), tf.zeros([5]), sample_weight=2.0), 15)
    self.assertEqual(m.total, 15)

  def test_reset_to_zero(self):
    m = counters.NumExamplesCounter()
    self.assertGreater(m(tf.zeros([10, 1]), tf.zeros([10])), 0)
    self.assertGreater(m.total, 0)
    m.reset_state()
    self.assertEqual(m.total, 0)


class NumBatchesCounterTest(tf.test.TestCase):

  def test_construct(self):
    m = counters.NumBatchesCounter()
    self.assertEqual(m.name, 'num_batches')
    self.assertTrue(m.stateful)
    self.assertEqual(m.dtype, tf.int64)
    self.assertLen(m.variables, 1)
    self.assertEqual(m.total, 0)
    m = counters.NumBatchesCounter('num_batches2')
    self.assertEqual(m.name, 'num_batches2')

  def test_update_without_sample_weight(self):
    m = counters.NumBatchesCounter()
    self.assertEqual(m(tf.zeros([10, 1]), tf.zeros([10])), 1)
    self.assertEqual(m.total, 1)
    self.assertEqual(m.update_state(tf.zeros([5, 1]), tf.zeros([5])), 2)
    self.assertEqual(m.total, 2)

  def test_update_with_sample_weight(self):
    m = counters.NumBatchesCounter()
    self.assertEqual(m(tf.zeros([10, 1]), tf.zeros([10]), sample_weight=0.5), 1)
    self.assertEqual(m.total, 1)
    self.assertEqual(
        m.update_state(tf.zeros([5, 1]), tf.zeros([5]), sample_weight=2.0), 2)
    self.assertEqual(m.total, 2)

  def test_reset_to_zero(self):
    m = counters.NumBatchesCounter()
    self.assertGreater(m(tf.zeros([10, 1]), tf.zeros([10])), 0)
    self.assertGreater(m.total, 0)
    m.reset_state()
    self.assertEqual(m.total, 0)


if __name__ == '__main__':
  tf.test.main()

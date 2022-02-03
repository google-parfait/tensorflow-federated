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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.learning.metrics import counters

NAMED_LABEL_TESTS = [
    ('single_label', np.zeros([10]), np.zeros([5, 1])),
    ('multi_label_list', [np.zeros([10])] * 2, [np.zeros([5, 1])] * 2),
    ('multi_label_dict', {
        'x1': np.zeros([10]),
        'y1': np.zeros([10])
    }, {
        'x2': np.zeros([5, 1]),
        'y2': np.zeros([5, 1])
    }),
]


class NumExamplesCounterTest(parameterized.TestCase, tf.test.TestCase):

  def test_construct(self):
    m = counters.NumExamplesCounter()
    self.assertEqual(m.name, 'num_examples')
    self.assertTrue(m.stateful)
    self.assertEqual(m.dtype, tf.int64)
    self.assertLen(m.variables, 1)
    self.assertEqual(m.total, 0)
    m = counters.NumExamplesCounter('num_examples2')
    self.assertEqual(m.name, 'num_examples2')

  @parameterized.named_parameters(NAMED_LABEL_TESTS)
  def test_update_without_sample_weight(self, batch1, batch2):
    m = counters.NumExamplesCounter()
    self.assertEqual(m(batch1, batch1), 10)
    self.assertEqual(m.total, 10)
    self.assertEqual(m.update_state(batch2, batch2), 15)
    self.assertEqual(m.total, 15)

  def test_update_computes_shape_of_first_input_arg(self):
    m = counters.NumExamplesCounter()
    self.assertEqual(m(tf.zeros([10, 1]), tf.zeros([15])), 10)
    self.assertEqual(m.total, 10)
    self.assertEqual(m.update_state(tf.zeros([5, 1]), tf.zeros([700])), 15)
    self.assertEqual(m.total, 15)

  @parameterized.named_parameters(NAMED_LABEL_TESTS)
  def test_update_with_sample_weight(self, batch1, batch2):
    m = counters.NumExamplesCounter()
    self.assertEqual(m(batch1, batch1), 10)
    self.assertEqual(m.total, 10)
    self.assertEqual(m.update_state(batch2, batch2), 15)
    self.assertEqual(m.total, 15)

  def test_reset_to_zero(self):
    m = counters.NumExamplesCounter()
    self.assertGreater(m(tf.zeros([10, 1]), tf.zeros([10])), 0)
    self.assertGreater(m.total, 0)
    m.reset_state()
    self.assertEqual(m.total, 0)


class NumBatchesCounterTest(parameterized.TestCase, tf.test.TestCase):

  def test_construct(self):
    m = counters.NumBatchesCounter()
    self.assertEqual(m.name, 'num_batches')
    self.assertTrue(m.stateful)
    self.assertEqual(m.dtype, tf.int64)
    self.assertLen(m.variables, 1)
    self.assertEqual(m.total, 0)
    m = counters.NumBatchesCounter('num_batches2')
    self.assertEqual(m.name, 'num_batches2')

  @parameterized.named_parameters(NAMED_LABEL_TESTS)
  def test_update_without_sample_weight(self, batch1, batch2):
    m = counters.NumBatchesCounter()
    self.assertEqual(m(batch1, batch1), 1)
    self.assertEqual(m.total, 1)
    self.assertEqual(m.update_state(batch2, batch2), 2)
    self.assertEqual(m.total, 2)

  @parameterized.named_parameters(NAMED_LABEL_TESTS)
  def test_update_with_sample_weight(self, batch1, batch2):
    m = counters.NumBatchesCounter()
    self.assertEqual(m(batch1, batch1), 1)
    self.assertEqual(m.total, 1)
    self.assertEqual(m.update_state(batch2, batch2), 2)
    self.assertEqual(m.total, 2)

  def test_reset_to_zero(self):
    m = counters.NumBatchesCounter()
    self.assertGreater(m(tf.zeros([10, 1]), tf.zeros([10])), 0)
    self.assertGreater(m.total, 0)
    m.reset_state()
    self.assertEqual(m.total, 0)


if __name__ == '__main__':
  tf.test.main()

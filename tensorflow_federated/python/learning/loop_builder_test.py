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

from collections.abc import Iterable
import itertools

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.learning import loop_builder

DATASET_REDUCE_OP = 'ReduceDataset'


def _get_op_names(graph_def: tf.compat.v1.GraphDef) -> Iterable[str]:
  all_nodes = itertools.chain(
      graph_def.node, *[f.node_def for f in graph_def.library.function]
  )
  return [n.name for n in all_nodes]


class DatasetReduceTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'dataset_iteration',
          loop_builder.LoopImplementation.DATASET_ITERATOR,
          loop_builder._for_iter_dataset_fn,
      ),
      (
          'dataset_reduce',
          loop_builder.LoopImplementation.DATASET_REDUCE,
          loop_builder._dataset_reduce_fn,
      ),
  )
  def test_build_training_loop(self, implementation, reduce_fn):
    dataset_reduce_fn = loop_builder.build_training_loop(implementation)
    self.assertIs(dataset_reduce_fn, reduce_fn)

  @parameterized.named_parameters(
      ('dataset_iteration', loop_builder._dataset_reduce_fn),
      ('dataset_reduce', loop_builder._for_iter_dataset_fn),
  )
  def test_reduction_math_is_correct(self, reduce_fn):
    with self.subTest('single_tensor'):
      ds = tf.data.Dataset.range(10, output_type=tf.int32)
      total_sum = reduce_fn(reduce_fn=lambda x, y: x + y, dataset=ds)
      self.assertEqual(total_sum, np.int32(45))
    with self.subTest('structure_of_tensors'):
      ds = tf.data.Dataset.range(10, output_type=tf.float32).map(
          lambda x: 0.1 * x
      )
      total_cnt, total_sum = reduce_fn(
          reduce_fn=lambda x, y: (x[0] + 1, x[1] + y),
          dataset=ds,
          initial_state_fn=lambda: (tf.constant(0.0), tf.constant(0.1)),
      )
      self.assertEqual(total_cnt, np.float32(10))
      self.assertEqual(total_sum, np.float32(4.6))

  def test_dataset_reduce_op_not_present_in_iterator(self):
    with tf.Graph().as_default() as graph:
      ds = tf.data.Dataset.range(10, output_type=tf.int32)
      tf.function(loop_builder._for_iter_dataset_fn)(
          reduce_fn=lambda x, y: x + y, dataset=ds
      )
      self.assertNotIn(DATASET_REDUCE_OP, _get_op_names(graph.as_graph_def()))

  def test_dataset_reduce_op_present_in_reduce(self):
    with tf.Graph().as_default() as graph:
      ds = tf.data.Dataset.range(10, output_type=tf.int32)
      tf.function(loop_builder._dataset_reduce_fn)(
          reduce_fn=lambda x, y: x + y, dataset=ds
      )
      self.assertIn(DATASET_REDUCE_OP, _get_op_names(graph.as_graph_def()))


if __name__ == '__main__':
  tf.test.main()

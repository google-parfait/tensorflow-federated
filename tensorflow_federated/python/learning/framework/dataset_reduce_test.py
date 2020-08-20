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

import itertools
from typing import Iterable

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.learning.framework import dataset_reduce

DATASET_REDUCE_OP = 'ReduceDataset'


def _get_op_names(graph_def: tf.compat.v1.GraphDef) -> Iterable[str]:
  all_nodes = itertools.chain(graph_def.node,
                              *[f.node_def for f in graph_def.library.function])
  return [n.name for n in all_nodes]


class DatasetReduceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('non-simulation', False, dataset_reduce._dataset_reduce_fn),
      ('simulation', True, dataset_reduce._for_iter_dataset_fn))
  def test_build_dataset_reduce_fn(self, simulation, reduce_fn):
    dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(simulation)
    self.assertIs(dataset_reduce, reduce_fn)
    ds = tf.data.Dataset.range(10, output_type=tf.int32)
    total_sum = dataset_reduce_fn(reduce_fn=lambda x, y: x + y, dataset=ds)
    self.assertEqual(total_sum, np.int32(45))

  @parameterized.named_parameters(
      ('non-simulation', False, dataset_reduce._dataset_reduce_fn),
      ('simulation', True, dataset_reduce._for_iter_dataset_fn))
  def test_build_dataset_reduce_fn_float(self, simulation, reduce_fn):
    dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(simulation)
    self.assertIs(dataset_reduce, reduce_fn)
    ds = tf.data.Dataset.range(
        10, output_type=tf.float32).map(lambda x: 0.1 * x)
    total_sum = dataset_reduce_fn(reduce_fn=lambda x, y: x + y, dataset=ds)
    self.assertEqual(total_sum, np.float32(4.5))

  @parameterized.named_parameters(
      ('non-simulation', False, dataset_reduce._dataset_reduce_fn),
      ('simulation', True, dataset_reduce._for_iter_dataset_fn))
  def test_build_dataset_reduce_fn_tuple(self, simulation, reduce_fn):
    dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(simulation)
    self.assertIs(dataset_reduce, reduce_fn)
    ds = tf.data.Dataset.range(
        10, output_type=tf.float32).map(lambda x: 0.1 * x)
    total_cnt, total_sum = dataset_reduce_fn(
        reduce_fn=lambda x, y: (x + 1, x + y),
        dataset=ds,
        initial_state_fn=lambda: (tf.constant(0), tf.constant(0.1)))
    self.assertEqual(total_cnt, np.float32(10))
    self.assertEqual(total_sum, np.float32(4.6))

  @parameterized.named_parameters(('non-simulation', False),
                                  ('simulation', True))
  def test_dataset_reduce_op_presence(self, simulation):
    with tf.Graph().as_default() as graph:
      dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(simulation)
      ds = tf.data.Dataset.range(10, output_type=tf.int32)
      dataset_reduce_fn(reduce_fn=lambda x, y: x + y, dataset=ds)
    if simulation:
      self.assertIn(DATASET_REDUCE_OP, _get_op_names(graph.as_graph_def()))
    else:
      self.assertNotIn(DATASET_REDUCE_OP, _get_op_names(graph.as_graph_def()))

# Lint as: python3
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
"""Tests for graph_spec.py."""

import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.tensorflow_libs import graph_spec


def _make_add_one_graph():
  with tf.Graph().as_default() as graph:
    input_val = tf.compat.v1.placeholder(tf.float32, name='input')
    const = tf.constant(1.0)
    out = tf.add(input_val, const)
  return graph, input_val.name, out.name


class GraphSpecTest(test.TestCase):

  def test_graph_spec_constructs_dummy_data(self):
    graph_def = _make_add_one_graph()[0].as_graph_def()
    init_op = 'init'
    in_names = ['in']
    out_names = ['out']
    x = graph_spec.GraphSpec(graph_def, init_op, in_names, out_names)
    self.assertIs(x.graph_def, graph_def)
    self.assertIs(x.init_op, init_op)
    self.assertIs(x.in_names, in_names)
    self.assertIs(x.out_names, out_names)

  def test_graph_spec_fails_no_graph_def(self):
    with self.assertRaises(TypeError):
      graph_spec.GraphSpec(None, 'test', ['test'], ['test'])

  def test_graph_spec_fails_bad_init_op(self):
    graph_def = _make_add_one_graph()[0].as_graph_def()
    with self.assertRaises(TypeError):
      graph_spec.GraphSpec(graph_def, 1, ['test'], ['test'])

  def test_graph_spec_succeeds_empty_init_op(self):
    graph_def = _make_add_one_graph()[0].as_graph_def()
    graph_spec.GraphSpec(graph_def, '', ['test'], ['test'])

  def test_graph_spec_fails_no_in_names(self):
    graph_def = _make_add_one_graph()[0].as_graph_def()
    with self.assertRaises(TypeError):
      graph_spec.GraphSpec(graph_def, 'test', None, ['test'])

  def test_graph_spec_fails_no_out_names(self):
    graph_def = _make_add_one_graph()[0].as_graph_def()
    with self.assertRaises(TypeError):
      graph_spec.GraphSpec(graph_def, 'test', ['test'], None)

  def test_graph_spec_fails_in_names_ints(self):
    graph_def = _make_add_one_graph()[0].as_graph_def()
    with self.assertRaises(TypeError):
      graph_spec.GraphSpec(graph_def, 'test', [1], ['test'])

  def test_graph_spec_fails_out_names_ints(self):
    graph_def = _make_add_one_graph()[0].as_graph_def()
    with self.assertRaises(TypeError):
      graph_spec.GraphSpec(graph_def, 'test', ['test'], [1])


if __name__ == '__main__':
  test.main()

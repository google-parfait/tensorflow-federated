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

import tensorflow.compat.v2 as tf

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


def _make_add_variable_number_graph(var_name=None):
  with tf.Graph().as_default() as graph:
    input_val = tf.compat.v1.placeholder(tf.float32, name='input')
    var = tf.Variable(initial_value=0.0, name=var_name, import_scope='')
    assign_op = var.assign_add(tf.constant(1.0))
    out = tf.add(input_val, assign_op)
  return graph, input_val.name, out.name


def _make_dataset_constructing_graph():
  with tf.Graph().as_default() as graph:
    d1 = tf.data.Dataset.range(5)
    v1 = tf.data.experimental.to_variant(d1)
  return graph, '', v1.name


def _make_manual_reduce_graph(dataset_construction_graph, return_element):
  with tf.Graph().as_default() as graph:
    v1 = tf.import_graph_def(
        dataset_construction_graph.as_graph_def(),
        return_elements=[return_element])[0]
    structure = tf.TensorSpec([], tf.int64)
    ds1 = tf.data.experimental.from_variant(v1, structure=structure)
    out = ds1.reduce(tf.constant(0, dtype=tf.int64), lambda x, y: x + y)
  return graph, '', out.name


class ToMetaGraphDefTest(test.TestCase):

  def test_graph_spec_to_meta_graph_def_simplest_case(self):
    graph, in_name, out_name = _make_add_one_graph()
    graph_def = graph.as_graph_def()
    init_op = None
    in_names = [in_name]
    out_names = [out_name]
    gs = graph_spec.GraphSpec(graph_def, init_op, in_names, out_names)
    metagraphdef = gs.to_meta_graph_def()
    self.assertIsInstance(metagraphdef, tf.compat.v1.MetaGraphDef)

  def test_meta_graph_def_runs_simplest_case(self):
    graph, in_name, out_name = _make_add_one_graph()
    graph_def = graph.as_graph_def()
    init_op = None
    in_names = [in_name]
    out_names = [out_name]
    gs = graph_spec.GraphSpec(graph_def, init_op, in_names, out_names)
    metagraphdef = gs.to_meta_graph_def()

    with tf.Graph().as_default() as g:
      tf.compat.v1.train.import_meta_graph(metagraphdef)

    with tf.compat.v1.Session(graph=g) as sess:
      should_be_one = sess.run(out_name, feed_dict={in_name: 0})

    self.assertEqual(should_be_one, 1.)

  def test_meta_graph_def_restores_and_runs_with_variables(self):
    graph, in_name, out_name = _make_add_variable_number_graph()

    with graph.as_default():
      init_op = tf.compat.v1.global_variables_initializer().name

    graph_def = graph.as_graph_def()
    in_names = [in_name]
    out_names = [out_name]
    gs = graph_spec.GraphSpec(graph_def, init_op, in_names, out_names)
    metagraphdef = gs.to_meta_graph_def()

    with tf.Graph().as_default() as g:
      tf.compat.v1.train.import_meta_graph(metagraphdef)
      restored_init_op = tf.group(
          *tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.INIT_OP)).name

    with tf.compat.v1.Session(graph=g) as sess:
      sess.run(restored_init_op)
      should_be_one = sess.run(out_name, feed_dict={in_name: 0})
      should_be_two = sess.run(out_name, feed_dict={in_name: 0})
      should_be_three = sess.run(out_name, feed_dict={in_name: 0})

    self.assertEqual(should_be_one, 1.)
    self.assertEqual(should_be_two, 2.)
    self.assertEqual(should_be_three, 3.)

  def test_meta_graph_def_restores_and_runs_with_datasets(self):
    dataset_graph, _, dataset_out_name = _make_dataset_constructing_graph()
    graph, _, out_name = _make_manual_reduce_graph(dataset_graph,
                                                   dataset_out_name)

    with graph.as_default():
      init_op = tf.compat.v1.global_variables_initializer().name

    graph_def = graph.as_graph_def()
    in_names = []
    out_names = [out_name]
    gs = graph_spec.GraphSpec(graph_def, init_op, in_names, out_names)
    metagraphdef = gs.to_meta_graph_def()

    with tf.Graph().as_default() as g:
      tf.compat.v1.train.import_meta_graph(metagraphdef)
      restored_init_op = tf.compat.v1.get_collection(
          tf.compat.v1.GraphKeys.INIT_OP)

    with tf.compat.v1.Session(graph=g) as sess:
      sess.run(restored_init_op)
      should_be_ten = sess.run(out_name)

    self.assertEqual(should_be_ten, 10)


if __name__ == '__main__':
  test.main()

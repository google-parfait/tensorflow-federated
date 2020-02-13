# Lint as: python3
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

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.tensorflow_libs import graph_merge

tf.compat.v1.enable_v2_behavior()


def _make_add_one_graph():
  with tf.Graph().as_default() as graph:
    input_val = tf.compat.v1.placeholder(tf.float32, name='input')
    const = tf.constant(1.0)
    out = tf.add(input_val, const)
  return graph, input_val.name, out.name


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


class GraphSpecTest(test.TestCase):

  def test_graph_spec_constructs_dummy_data(self):
    graph_def = _make_add_one_graph()[0].as_graph_def()
    init_op = 'init'
    in_names = ['in']
    out_names = ['out']
    x = graph_merge.GraphSpec(graph_def, init_op, in_names, out_names)
    self.assertIs(x.graph_def, graph_def)
    self.assertIs(x.init_op, init_op)
    self.assertIs(x.in_names, in_names)
    self.assertIs(x.out_names, out_names)

  def test_graph_spec_fails_no_graph_def(self):
    with self.assertRaises(TypeError):
      graph_merge.GraphSpec(None, 'test', ['test'], ['test'])

  def test_graph_spec_fails_bad_init_op(self):
    graph_def = _make_add_one_graph()[0].as_graph_def()
    with self.assertRaises(TypeError):
      graph_merge.GraphSpec(graph_def, 1, ['test'], ['test'])

  def test_graph_spec_succeeds_empty_init_op(self):
    graph_def = _make_add_one_graph()[0].as_graph_def()
    graph_merge.GraphSpec(graph_def, '', ['test'], ['test'])

  def test_graph_spec_fails_no_in_names(self):
    graph_def = _make_add_one_graph()[0].as_graph_def()
    with self.assertRaises(TypeError):
      graph_merge.GraphSpec(graph_def, 'test', None, ['test'])

  def test_graph_spec_fails_no_out_names(self):
    graph_def = _make_add_one_graph()[0].as_graph_def()
    with self.assertRaises(TypeError):
      graph_merge.GraphSpec(graph_def, 'test', ['test'], None)


class ConcatenateInputsAndOutputsTest(test.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      graph_merge.concatenate_inputs_and_outputs(None)

  def test_raises_on_non_iterable(self):
    with self.assertRaises(TypeError):
      graph_merge.concatenate_inputs_and_outputs(1)

  def test_concatenate_inputs_and_outputs_two_add_one_graphs(self):
    graph1, input_name_1, output_name_1 = _make_add_one_graph()
    graph2, input_name_2, output_name_2 = _make_add_one_graph()
    with graph1.as_default():
      init_op_name_1 = tf.compat.v1.global_variables_initializer().name
    with graph2.as_default():
      init_op_name_2 = tf.compat.v1.global_variables_initializer().name
    graph_spec_1 = graph_merge.GraphSpec(graph1.as_graph_def(), init_op_name_1,
                                         [input_name_1], [output_name_1])
    graph_spec_2 = graph_merge.GraphSpec(graph2.as_graph_def(), init_op_name_2,
                                         [input_name_2], [output_name_2])
    arg_list = [graph_spec_1, graph_spec_2]
    merged_graph, init_op_name, in_name_maps, out_name_maps = graph_merge.concatenate_inputs_and_outputs(
        arg_list)

    with merged_graph.as_default():
      with tf.compat.v1.Session() as sess:
        sess.run(init_op_name)
        outputs = sess.run(
            [out_name_maps[0][output_name_1], out_name_maps[1][output_name_2]],
            feed_dict={
                in_name_maps[0][input_name_1]: 1.0,
                in_name_maps[1][input_name_2]: 2.0
            })

    self.assertAllClose(outputs, np.array([2., 3.]))

  def test_concatenate_inputs_and_outputs_three_add_one_graphs(self):
    graph1, input_name_1, output_name_1 = _make_add_one_graph()
    graph2, input_name_2, output_name_2 = _make_add_one_graph()
    graph3, input_name_3, output_name_3 = _make_add_one_graph()
    with graph1.as_default():
      init_op_name_1 = tf.compat.v1.global_variables_initializer().name
    with graph2.as_default():
      init_op_name_2 = tf.compat.v1.global_variables_initializer().name
    with graph3.as_default():
      init_op_name_3 = tf.compat.v1.global_variables_initializer().name
    graph_spec_1 = graph_merge.GraphSpec(graph1.as_graph_def(), init_op_name_1,
                                         [input_name_1], [output_name_1])
    graph_spec_2 = graph_merge.GraphSpec(graph2.as_graph_def(), init_op_name_2,
                                         [input_name_2], [output_name_2])
    graph_spec_3 = graph_merge.GraphSpec(graph3.as_graph_def(), init_op_name_3,
                                         [input_name_3], [output_name_3])
    arg_list = [graph_spec_1, graph_spec_2, graph_spec_3]
    merged_graph, init_op_name, in_name_maps, out_name_maps = graph_merge.concatenate_inputs_and_outputs(
        arg_list)

    with merged_graph.as_default():
      with tf.compat.v1.Session() as sess:
        sess.run(init_op_name)
        outputs = sess.run(
            [
                out_name_maps[0][output_name_1],
                out_name_maps[1][output_name_2], out_name_maps[2][output_name_3]
            ],
            feed_dict={
                in_name_maps[0][input_name_1]: 1.0,
                in_name_maps[1][input_name_2]: 2.0,
                in_name_maps[2][input_name_3]: 3.0
            })

    self.assertAllClose(outputs, np.array([2., 3., 4.]))

  def test_concatenate_inputs_and_outputs_no_arg_graphs(self):
    graph1 = tf.Graph()
    with graph1.as_default():
      out1 = tf.constant(1.0)
      init_op_name_1 = tf.compat.v1.global_variables_initializer().name
    graph2 = tf.Graph()
    with graph2.as_default():
      out2 = tf.constant(2.0)
      init_op_name_2 = tf.compat.v1.global_variables_initializer().name

    graph_spec_1 = graph_merge.GraphSpec(graph1.as_graph_def(), init_op_name_1,
                                         [], [out1.name])
    graph_spec_2 = graph_merge.GraphSpec(graph2.as_graph_def(), init_op_name_2,
                                         [], [out2.name])
    arg_list = [graph_spec_1, graph_spec_2]
    merged_graph, init_op_name, _, out_name_maps = graph_merge.concatenate_inputs_and_outputs(
        arg_list)

    with merged_graph.as_default():
      with tf.compat.v1.Session() as sess:
        sess.run(init_op_name)
        outputs = sess.run(
            [out_name_maps[0][out1.name], out_name_maps[1][out2.name]])

    self.assertAllClose(outputs, np.array([1., 2.]))

  def test_concatenate_inputs_and_outputs_no_init_op_graphs(self):
    graph1, input_name_1, output_name_1 = _make_add_one_graph()
    graph2, input_name_2, output_name_2 = _make_add_one_graph()
    graph_spec_1 = graph_merge.GraphSpec(graph1.as_graph_def(), None,
                                         [input_name_1], [output_name_1])
    graph_spec_2 = graph_merge.GraphSpec(graph2.as_graph_def(), None,
                                         [input_name_2], [output_name_2])
    arg_list = [graph_spec_1, graph_spec_2]
    merged_graph, init_op_name, in_name_maps, out_name_maps = graph_merge.concatenate_inputs_and_outputs(
        arg_list)

    with merged_graph.as_default():
      with tf.compat.v1.Session() as sess:
        sess.run(init_op_name)
        outputs = sess.run(
            [out_name_maps[0][output_name_1], out_name_maps[1][output_name_2]],
            feed_dict={
                in_name_maps[0][input_name_1]: 1.0,
                in_name_maps[1][input_name_2]: 2.0
            })

    self.assertAllClose(outputs, np.array([2., 3.]))

  def test_concatenate_inputs_and_outputs_two_add_variable_number_graphs(self):
    graph1, input_name_1, output_name_1 = _make_add_variable_number_graph()
    graph2, input_name_2, output_name_2 = _make_add_variable_number_graph()
    with graph1.as_default():
      init_op_name_1 = tf.compat.v1.global_variables_initializer().name
    with graph2.as_default():
      init_op_name_2 = tf.compat.v1.global_variables_initializer().name
    graph_spec_1 = graph_merge.GraphSpec(graph1.as_graph_def(), init_op_name_1,
                                         [input_name_1], [output_name_1])
    graph_spec_2 = graph_merge.GraphSpec(graph2.as_graph_def(), init_op_name_2,
                                         [input_name_2], [output_name_2])
    arg_list = [graph_spec_1, graph_spec_2]
    merged_graph, init_op_name, in_name_maps, out_name_maps = graph_merge.concatenate_inputs_and_outputs(
        arg_list)

    with merged_graph.as_default():
      with tf.compat.v1.Session() as sess:
        sess.run(init_op_name)
        outputs_1 = sess.run(
            [out_name_maps[0][output_name_1], out_name_maps[1][output_name_2]],
            feed_dict={
                in_name_maps[0][input_name_1]: 1.0,
                in_name_maps[1][input_name_2]: 2.0
            })
        outputs_2 = sess.run(
            [out_name_maps[0][output_name_1], out_name_maps[1][output_name_2]],
            feed_dict={
                in_name_maps[0][input_name_1]: 1.0,
                in_name_maps[1][input_name_2]: 2.0
            })
        outputs_3 = sess.run(
            [out_name_maps[0][output_name_1], out_name_maps[1][output_name_2]],
            feed_dict={
                in_name_maps[0][input_name_1]: 1.0,
                in_name_maps[1][input_name_2]: 2.0
            })
    self.assertAllClose(outputs_1, [2., 3.])
    self.assertAllClose(outputs_2, [3., 4.])
    self.assertAllClose(outputs_3, [4., 5.])

  def test_concatenate_inputs_and_outputs_with_dataset_wires_correctly(self):
    dataset_graph, _, dataset_out_name = _make_dataset_constructing_graph()
    graph_1, _, out_name_1 = _make_manual_reduce_graph(dataset_graph,
                                                       dataset_out_name)
    graph_2, _, out_name_2 = _make_manual_reduce_graph(dataset_graph,
                                                       dataset_out_name)
    with graph_1.as_default():
      init_op_name_1 = tf.compat.v1.global_variables_initializer().name
    with graph_2.as_default():
      init_op_name_2 = tf.compat.v1.global_variables_initializer().name
    graph_spec_1 = graph_merge.GraphSpec(graph_1.as_graph_def(), init_op_name_1,
                                         [], [out_name_1])
    graph_spec_2 = graph_merge.GraphSpec(graph_2.as_graph_def(), init_op_name_2,
                                         [], [out_name_2])
    arg_list = [graph_spec_1, graph_spec_2]
    merged_graph, init_op_name, _, out_name_maps = graph_merge.concatenate_inputs_and_outputs(
        arg_list)

    with merged_graph.as_default():
      with tf.compat.v1.Session() as sess:
        sess.run(init_op_name)
        tens = sess.run(
            [out_name_maps[0][out_name_1], out_name_maps[1][out_name_2]])
    self.assertEqual(tens, [10, 10])


class ComposeGraphSpecTest(test.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      graph_merge.compose_graph_specs(None)

  def test_raises_on_graph_spec_set(self):
    graph1, input_name_1, output_name_1 = _make_add_one_graph()
    graph_spec_1 = graph_merge.GraphSpec(graph1.as_graph_def(), '',
                                         [input_name_1], [output_name_1])
    with self.assertRaises(TypeError):
      graph_merge.compose_graph_specs(set(graph_spec_1))

  def test_raises_on_list_of_ints(self):
    with self.assertRaises(TypeError):
      graph_merge.compose_graph_specs([0, 1])

  def test_compose_no_input_graphs_raises(self):
    graph1 = tf.Graph()
    with graph1.as_default():
      out1 = tf.constant(1.0)
      init_op_name_1 = tf.compat.v1.global_variables_initializer().name
    graph2 = tf.Graph()
    with graph2.as_default():
      out2 = tf.constant(2.0)
      init_op_name_2 = tf.compat.v1.global_variables_initializer().name

    graph_spec_1 = graph_merge.GraphSpec(graph1.as_graph_def(), init_op_name_1,
                                         [], [out1.name])
    graph_spec_2 = graph_merge.GraphSpec(graph2.as_graph_def(), init_op_name_2,
                                         [], [out2.name])
    arg_list = [graph_spec_1, graph_spec_2]
    with self.assertRaisesRegex(ValueError, 'mismatch'):
      graph_merge.compose_graph_specs(arg_list)

  def test_compose_two_add_one_graphs_adds_two(self):
    graph1, input_name_1, output_name_1 = _make_add_one_graph()
    graph2, input_name_2, output_name_2 = _make_add_one_graph()
    with graph1.as_default():
      init_op_name_1 = tf.compat.v1.global_variables_initializer().name
    with graph2.as_default():
      init_op_name_2 = tf.compat.v1.global_variables_initializer().name
    graph_spec_1 = graph_merge.GraphSpec(graph1.as_graph_def(), init_op_name_1,
                                         [input_name_1], [output_name_1])
    graph_spec_2 = graph_merge.GraphSpec(graph2.as_graph_def(), init_op_name_2,
                                         [input_name_2], [output_name_2])
    arg_list = [graph_spec_1, graph_spec_2]
    composed_graph, init_op_name, in_name_map, out_name_map = graph_merge.compose_graph_specs(
        arg_list)

    with composed_graph.as_default():
      with tf.compat.v1.Session() as sess:
        sess.run(init_op_name)
        outputs = sess.run(
            out_name_map[output_name_2],
            feed_dict={
                in_name_map[input_name_1]: 0.0,
            })

    self.assertAllClose(outputs, np.array(2.))

  def test_composition_happens_in_mathematical_composition_order(self):
    graph1, input_name_1, output_name_1 = _make_add_one_graph()

    def _make_cast_to_int_graph():
      with tf.Graph().as_default() as graph:
        input_val = tf.compat.v1.placeholder(tf.float32, name='input')
        out = tf.cast(input_val, tf.int32)
      return graph, input_val.name, out.name

    graph2, input_name_2, output_name_2 = _make_cast_to_int_graph()

    with graph1.as_default():
      init_op_name_1 = tf.compat.v1.global_variables_initializer().name
    with graph2.as_default():
      init_op_name_2 = tf.compat.v1.global_variables_initializer().name
    graph_spec_1 = graph_merge.GraphSpec(graph1.as_graph_def(), init_op_name_1,
                                         [input_name_1], [output_name_1])
    graph_spec_2 = graph_merge.GraphSpec(graph2.as_graph_def(), init_op_name_2,
                                         [input_name_2], [output_name_2])
    arg_list = [graph_spec_2, graph_spec_1]

    composed_graph, _, in_name_map, out_name_map = graph_merge.compose_graph_specs(
        arg_list)

    with composed_graph.as_default():
      with tf.compat.v1.Session() as sess:
        outputs = sess.run(
            out_name_map[output_name_2],
            feed_dict={
                in_name_map[input_name_1]: 0.0,
            })

    self.assertEqual(outputs, 1)

    with self.assertRaises(ValueError):
      graph_merge.compose_graph_specs(list(reversed(arg_list)))

  def test_compose_three_add_one_graphs_adds_three(self):
    graph1, input_name_1, output_name_1 = _make_add_one_graph()
    graph2, input_name_2, output_name_2 = _make_add_one_graph()
    graph3, input_name_3, output_name_3 = _make_add_one_graph()
    with graph1.as_default():
      init_op_name_1 = tf.compat.v1.global_variables_initializer().name
    with graph2.as_default():
      init_op_name_2 = tf.compat.v1.global_variables_initializer().name
    with graph3.as_default():
      init_op_name_3 = tf.compat.v1.global_variables_initializer().name
    graph_spec_1 = graph_merge.GraphSpec(graph1.as_graph_def(), init_op_name_1,
                                         [input_name_1], [output_name_1])
    graph_spec_2 = graph_merge.GraphSpec(graph2.as_graph_def(), init_op_name_2,
                                         [input_name_2], [output_name_2])
    graph_spec_3 = graph_merge.GraphSpec(graph3.as_graph_def(), init_op_name_3,
                                         [input_name_3], [output_name_3])
    arg_list = [graph_spec_1, graph_spec_2, graph_spec_3]
    composed_graph, init_op_name, in_name_map, out_name_map = graph_merge.compose_graph_specs(
        arg_list)

    with composed_graph.as_default():
      with tf.compat.v1.Session() as sess:
        sess.run(init_op_name)
        outputs = sess.run(
            out_name_map[output_name_3],
            feed_dict={
                in_name_map[input_name_1]: 0.0,
            })

    self.assertAllClose(outputs, np.array(3.))

  def test_compose_two_add_variable_number_graphs_executes_correctly(self):
    graph1, input_name_1, output_name_1 = _make_add_variable_number_graph()
    graph2, input_name_2, output_name_2 = _make_add_variable_number_graph()
    with graph1.as_default():
      init_op_name_1 = tf.compat.v1.global_variables_initializer().name
    with graph2.as_default():
      init_op_name_2 = tf.compat.v1.global_variables_initializer().name
    graph_spec_1 = graph_merge.GraphSpec(graph1.as_graph_def(), init_op_name_1,
                                         [input_name_1], [output_name_1])
    graph_spec_2 = graph_merge.GraphSpec(graph2.as_graph_def(), init_op_name_2,
                                         [input_name_2], [output_name_2])
    arg_list = [graph_spec_1, graph_spec_2]
    composed_graph, init_op_name, in_name_map, out_name_map = graph_merge.compose_graph_specs(
        arg_list)

    with composed_graph.as_default():
      with tf.compat.v1.Session() as sess:
        sess.run(init_op_name)
        output_one = sess.run(
            out_name_map[output_name_2],
            feed_dict={
                in_name_map[input_name_1]: 0.0,
            })
        output_two = sess.run(
            out_name_map[output_name_2],
            feed_dict={
                in_name_map[input_name_1]: 0.0,
            })
        output_three = sess.run(
            out_name_map[output_name_2],
            feed_dict={
                in_name_map[input_name_1]: 0.0,
            })

    self.assertAllClose(output_one, np.array(2.))
    self.assertAllClose(output_two, np.array(4.))
    self.assertAllClose(output_three, np.array(6.))

  def test_compose_with_dataset_wires_correctly(self):
    with tf.Graph().as_default() as dataset_graph:
      d1 = tf.data.Dataset.range(5)
      v1 = tf.data.experimental.to_variant(d1)

    ds_out_name = v1.name
    variant_type = v1.dtype

    with tf.Graph().as_default() as reduce_graph:
      variant = tf.compat.v1.placeholder(variant_type)
      structure = tf.TensorSpec([], tf.int64)
      ds1 = tf.data.experimental.from_variant(variant, structure=structure)
      out = ds1.reduce(tf.constant(0, dtype=tf.int64), lambda x, y: x + y)

    ds_in_name = variant.name
    reduce_out_name = out.name

    with dataset_graph.as_default():
      init_op_name_1 = tf.compat.v1.global_variables_initializer().name
    with reduce_graph.as_default():
      init_op_name_2 = tf.compat.v1.global_variables_initializer().name
    dataset_graph_spec = graph_merge.GraphSpec(dataset_graph.as_graph_def(),
                                               init_op_name_1, [],
                                               [ds_out_name])
    reduce_graph_spec = graph_merge.GraphSpec(reduce_graph.as_graph_def(),
                                              init_op_name_2, [ds_in_name],
                                              [reduce_out_name])
    arg_list = [reduce_graph_spec, dataset_graph_spec]
    composed_graph, _, _, out_name_map = graph_merge.compose_graph_specs(
        arg_list)

    with composed_graph.as_default():
      with tf.compat.v1.Session() as sess:
        ten = sess.run(out_name_map[reduce_out_name])
    self.assertEqual(ten, 10)


if __name__ == '__main__':
  test.main()

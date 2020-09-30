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
"""Tests for graph_optimizations.py."""

import tensorflow as tf

from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.tensorflow_libs import graph_optimizations
from tensorflow_federated.python.tensorflow_libs import graph_spec


def _make_redundant_add_one_graph():
  with tf.Graph().as_default() as graph:
    input_val = tf.compat.v1.placeholder(tf.float32, name='input')
    const = tf.constant(1.0)
    const1 = tf.constant(1.0)
    const2 = tf.constant(1.0)
    const3 = tf.constant(1.0)
    const4 = tf.constant(1.0)
    _ = const1 + const2 + const3 + const4
    out = tf.add(input_val, const)
  return graph, input_val.name, out.name


def _make_foldable_add_variable_number_graph(var_name=None):
  with tf.Graph().as_default() as graph:
    input_val = tf.compat.v1.placeholder(tf.float32, name='input')
    const1 = tf.constant(1.0)
    const2 = tf.constant(1.0)
    const3 = tf.constant(1.0)
    const4 = tf.constant(1.0)
    const5 = const1 + const2 + const3 + const4
    var = tf.Variable(initial_value=0.0, name=var_name, import_scope='')
    assign_op = var.assign_add(const5)
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


class GraphOptTest(test_utils.TestCase):

  def test_reduces_bytesize_for_simple_graphdef(self):
    graph, in_name, out_name = _make_redundant_add_one_graph()
    graph_def = graph.as_graph_def()
    init_op = None
    in_names = [in_name]
    out_names = [out_name]
    gs = graph_spec.GraphSpec(graph_def, init_op, in_names, out_names)
    config_proto = tf.compat.v1.ConfigProto()
    opt_graph_spec = graph_optimizations.optimize_graph_spec(gs, config_proto)
    self.assertIsInstance(opt_graph_spec, graph_spec.GraphSpec)
    self.assertLess(opt_graph_spec.graph_def.ByteSize(), graph_def.ByteSize())

  def test_semantic_equivalence_for_simple_graphdef(self):
    graph, in_name, out_name = _make_redundant_add_one_graph()
    graph_def = graph.as_graph_def()
    init_op = None
    in_names = [in_name]
    out_names = [out_name]
    gs = graph_spec.GraphSpec(graph_def, init_op, in_names, out_names)
    config_proto = tf.compat.v1.ConfigProto()
    opt_graph_spec = graph_optimizations.optimize_graph_spec(gs, config_proto)

    with tf.Graph().as_default() as orig_graph:
      tf.graph_util.import_graph_def(gs.graph_def, name='')

    with tf.compat.v1.Session(graph=orig_graph) as sess:
      orig_out = sess.run(gs.out_names, feed_dict={x: 1 for x in gs.in_names})

    with tf.Graph().as_default() as new_graph:
      tf.graph_util.import_graph_def(opt_graph_spec.graph_def, name='')

    with tf.compat.v1.Session(graph=new_graph) as sess:
      new_out = sess.run(
          opt_graph_spec.out_names,
          feed_dict={x: 1 for x in opt_graph_spec.in_names})

    self.assertEqual(new_out, orig_out)

  def test_reduces_bytesize_for_dataset_reduction(self):
    ds_graph, _, out = _make_dataset_constructing_graph()
    graph, _, out_name = _make_manual_reduce_graph(ds_graph, out)
    with graph.as_default():
      init_op = tf.compat.v1.global_variables_initializer().name
    graph_def = graph.as_graph_def()
    in_names = []
    out_names = [out_name]
    gs = graph_spec.GraphSpec(graph_def, init_op, in_names, out_names)
    config_proto = tf.compat.v1.ConfigProto()
    opt_graph_spec = graph_optimizations.optimize_graph_spec(gs, config_proto)
    self.assertIsInstance(opt_graph_spec, graph_spec.GraphSpec)
    self.assertLess(opt_graph_spec.graph_def.ByteSize(), graph_def.ByteSize())

  def test_semantic_equivalence_for_reduction(self):
    ds_graph, _, out = _make_dataset_constructing_graph()
    graph, _, out_name = _make_manual_reduce_graph(ds_graph, out)

    with graph.as_default():
      init_op = tf.compat.v1.global_variables_initializer().name

    graph_def = graph.as_graph_def()
    in_names = []
    out_names = [out_name]
    gs = graph_spec.GraphSpec(graph_def, init_op, in_names, out_names)
    config_proto = tf.compat.v1.ConfigProto()
    opt_graph_spec = graph_optimizations.optimize_graph_spec(gs, config_proto)

    with tf.Graph().as_default() as orig_graph:
      tf.graph_util.import_graph_def(gs.graph_def, name='')

    with tf.compat.v1.Session(graph=orig_graph) as sess:
      sess.run(gs.init_op)
      orig_out = sess.run(gs.out_names, feed_dict={x: 1 for x in gs.in_names})

    with tf.Graph().as_default() as new_graph:
      tf.graph_util.import_graph_def(opt_graph_spec.graph_def, name='')

    with tf.compat.v1.Session(graph=new_graph) as new_sess:
      new_sess.run(opt_graph_spec.init_op)
      new_out = new_sess.run(
          opt_graph_spec.out_names,
          feed_dict={x: 1 for x in opt_graph_spec.in_names})

    self.assertEqual(new_out, orig_out)

  def test_reduces_bytesize_for_foldable_graphdef_with_variables(self):
    graph, in_name, out_name = _make_foldable_add_variable_number_graph()
    with graph.as_default():
      init_op = tf.compat.v1.global_variables_initializer().name
    graph_def = graph.as_graph_def()

    orig_constants_1 = []
    for node in graph_def.node:
      if node.op == 'Const':
        for float_val in node.attr['value'].tensor.float_val:
          if float_val == 1.:
            orig_constants_1.append(node)

    in_names = [in_name]
    out_names = [out_name]
    gs = graph_spec.GraphSpec(graph_def, init_op, in_names, out_names)
    config_proto = tf.compat.v1.ConfigProto()
    opt_graph_spec = graph_optimizations.optimize_graph_spec(gs, config_proto)

    opt_constants_1 = []
    for node in opt_graph_spec.graph_def.node:
      if node.op == 'Const':
        for float_val in node.attr['value'].tensor.float_val:
          if float_val == 1.:
            opt_constants_1.append(node)

    self.assertIsInstance(opt_graph_spec, graph_spec.GraphSpec)
    self.assertLess(opt_graph_spec.graph_def.ByteSize(), graph_def.ByteSize())
    self.assertGreater(len(orig_constants_1), 1)
    self.assertLess(len(opt_constants_1), len(orig_constants_1))

  def test_semantic_equivalence_for_graphdef_with_variables(self):
    graph, in_name, out_name = _make_foldable_add_variable_number_graph()
    with graph.as_default():
      init_op = tf.compat.v1.global_variables_initializer().name
    graph_def = graph.as_graph_def()
    in_names = [in_name]
    out_names = [out_name]
    gs = graph_spec.GraphSpec(graph_def, init_op, in_names, out_names)
    config_proto = tf.compat.v1.ConfigProto()
    opt_graph_spec = graph_optimizations.optimize_graph_spec(gs, config_proto)

    with tf.Graph().as_default() as orig_graph:
      tf.graph_util.import_graph_def(gs.graph_def, name='')

    with tf.compat.v1.Session(graph=orig_graph) as sess:
      sess.run(gs.init_op)
      orig_out = sess.run(gs.out_names, feed_dict={x: 1 for x in gs.in_names})

    with tf.Graph().as_default() as new_graph:
      tf.graph_util.import_graph_def(opt_graph_spec.graph_def, name='')

    with tf.compat.v1.Session(graph=new_graph) as new_sess:
      new_sess.run(opt_graph_spec.init_op)
      new_out = new_sess.run(
          opt_graph_spec.out_names,
          feed_dict={x: 1 for x in opt_graph_spec.in_names})

    self.assertEqual(new_out, orig_out)

  def test_reduces_graph_size_in_function_lib(self):

    class StateHolder:
      pass

    obj = StateHolder()
    obj.variable = None

    @tf.function
    def foo(x):
      if obj.variable is None:
        obj.variable = tf.Variable(initial_value=0.)
        obj.variable.assign_add(x)
      return obj.variable.read_value()

    with tf.Graph().as_default() as g:
      x = tf.compat.v1.placeholder(shape=[], dtype=tf.float32)
      y = foo(x)
      init_op = tf.compat.v1.global_variables_initializer()

    graph_def = g.as_graph_def()
    in_name = x.name
    out_name = y.name
    init_op_name = init_op.name

    in_names = [in_name]
    out_names = [out_name]
    gs = graph_spec.GraphSpec(graph_def, init_op_name, in_names, out_names)
    config_proto = tf.compat.v1.ConfigProto()
    opt_graph_spec = graph_optimizations.optimize_graph_spec(gs, config_proto)

    self.assertIsInstance(opt_graph_spec, graph_spec.GraphSpec)
    self.assertLess(opt_graph_spec.graph_def.ByteSize(), graph_def.ByteSize())

  def test_semantic_equivalence_for_graphdef_with_function(self):

    class StateHolder:
      pass

    obj = StateHolder()
    obj.variable = None

    @tf.function
    def foo(x):
      if obj.variable is None:
        obj.variable = tf.Variable(initial_value=0.)
        obj.variable.assign_add(x)
      return obj.variable.read_value()

    with tf.Graph().as_default() as g:
      x = tf.compat.v1.placeholder(shape=[], dtype=tf.float32)
      y = foo(x)
      init_op = tf.compat.v1.global_variables_initializer()

    graph_def = g.as_graph_def()
    in_name = x.name
    out_name = y.name
    init_op_name = init_op.name

    in_names = [in_name]
    out_names = [out_name]
    gs = graph_spec.GraphSpec(graph_def, init_op_name, in_names, out_names)
    config_proto = tf.compat.v1.ConfigProto()
    opt_graph_spec = graph_optimizations.optimize_graph_spec(gs, config_proto)

    with tf.Graph().as_default() as orig_graph:
      tf.graph_util.import_graph_def(gs.graph_def, name='')

    with tf.compat.v1.Session(graph=orig_graph) as sess:
      sess.run(gs.init_op)
      orig_out = sess.run(gs.out_names, feed_dict={x: 1 for x in gs.in_names})

    with tf.Graph().as_default() as new_graph:
      tf.graph_util.import_graph_def(opt_graph_spec.graph_def, name='')

    with tf.compat.v1.Session(graph=new_graph) as new_sess:
      new_sess.run(opt_graph_spec.init_op)
      new_out = new_sess.run(
          opt_graph_spec.out_names,
          feed_dict={x: 1 for x in opt_graph_spec.in_names})

    self.assertEqual(new_out, orig_out)


if __name__ == '__main__':
  test_utils.main()

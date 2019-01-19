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
"""Tests for computations.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import computation_wrapper_instances


class ComputationWrapperInstancesTest(test.TestCase):

  def test_tf_wrapper_with_one_op_py_func(self):

    @computation_wrapper_instances.tensorflow_wrapper(tf.int32)
    def foo(x):
      return x > 10

    self.assertEqual(str(foo.type_signature), '(int32 -> bool)')

    # TODO(b/113112885): Remove this protected member access once the part of
    # the infrastructure that deals with invoking functions is present. At this
    # point, extracting the proto from within 'foo' is the only way to test the
    # wrapper works as intended.
    comp = foo._computation_proto  # pylint: disable=protected-access

    self.assertEqual(comp.WhichOneof('computation'), 'tensorflow')
    x = tf.placeholder(tf.int32)
    result = tf.import_graph_def(
        comp.tensorflow.graph_def,
        {comp.tensorflow.parameter.tensor.tensor_name: x},
        [comp.tensorflow.result.tensor.tensor_name])
    self.assertEqual(
        list(tf.Session().run(result, feed_dict={x: n})
             for n in [1, 20, 5, 10, 30]),
        [[False], [True], [False], [False], [True]])

  def test_tf_wrapper_with_tf_add(self):
    foo = computation_wrapper_instances.tensorflow_wrapper(
        tf.add, (tf.int32, tf.int32))
    self.assertEqual(str(foo.type_signature), '(<int32,int32> -> int32)')

    # TODO(b/113112885): Remove this protected member access as noted above.
    comp = foo._computation_proto  # pylint: disable=protected-access

    self.assertEqual(comp.WhichOneof('computation'), 'tensorflow')
    x = tf.placeholder(tf.int32)
    y = tf.placeholder(tf.int32)
    result = tf.import_graph_def(
        comp.tensorflow.graph_def, {
            comp.tensorflow.parameter.tuple.element[0].tensor.tensor_name: x,
            comp.tensorflow.parameter.tuple.element[1].tensor.tensor_name: y
        }, [comp.tensorflow.result.tensor.tensor_name])
    with self.session() as sess:
      results = [
          sess.run(result, feed_dict={x: n, y: 3}) for n in [1, 20, 5, 10, 30]
      ]
      self.assertEqual(results, [[4], [23], [8], [13], [33]])

  def test_federated_computation_wrapper(self):

    @computation_wrapper_instances.federated_computation_wrapper(
        (computation_types.FunctionType(tf.int32, tf.int32), tf.int32))
    def foo(f, x):
      return f(f(x))

    self.assertIsInstance(foo, computation_impl.ComputationImpl)
    self.assertEqual(
        str(foo.type_signature), '(<(int32 -> int32),int32> -> int32)')

    # TODO(b/113112885): Remove this protected member access as noted above.
    comp = foo._computation_proto

    building_block = (
        computation_building_blocks.ComputationBuildingBlock.from_proto(comp))
    self.assertEqual(str(building_block), '(arg -> arg[0](arg[0](arg[1])))')


if __name__ == '__main__':
  test.main()

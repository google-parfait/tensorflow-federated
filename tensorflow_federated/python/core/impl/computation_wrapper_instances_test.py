# Lint as: python3
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

import tensorflow as tf

from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import computation_wrapper_instances
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import placement_literals


class ComputationWrapperInstancesTest(test.TestCase):

  @test.graph_mode_test
  def test_tf_wrapper_with_one_op_py_fn(self):

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
    x = tf.compat.v1.placeholder(tf.int32)
    result = tf.import_graph_def(
        serialization_utils.unpack_graph_def(comp.tensorflow.graph_def),
        {comp.tensorflow.parameter.tensor.tensor_name: x},
        [comp.tensorflow.result.tensor.tensor_name])
    self.assertEqual(
        list(tf.compat.v1.Session().run(result, feed_dict={x: n})
             for n in [1, 20, 5, 10, 30]),
        [[False], [True], [False], [False], [True]])

  @test.graph_mode_test
  def test_tf_wrapper_with_tf_add(self):
    foo = computation_wrapper_instances.tensorflow_wrapper(
        tf.add, (tf.int32, tf.int32))
    self.assertEqual(str(foo.type_signature), '(<int32,int32> -> int32)')

    # TODO(b/113112885): Remove this protected member access as noted above.
    comp = foo._computation_proto  # pylint: disable=protected-access

    self.assertEqual(comp.WhichOneof('computation'), 'tensorflow')
    x = tf.compat.v1.placeholder(tf.int32)
    y = tf.compat.v1.placeholder(tf.int32)
    result = tf.import_graph_def(
        serialization_utils.unpack_graph_def(comp.tensorflow.graph_def), {
            comp.tensorflow.parameter.tuple.element[0].tensor.tensor_name: x,
            comp.tensorflow.parameter.tuple.element[1].tensor.tensor_name: y
        }, [comp.tensorflow.result.tensor.tensor_name])
    with self.session() as sess:

      def _run(n):
        return sess.run(result, feed_dict={x: n, y: 3})

      results = [_run(n) for n in [1, 20, 5, 10, 30]]
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

    building_block = (building_blocks.ComputationBuildingBlock.from_proto(comp))
    self.assertEqual(
        str(building_block),
        '(FEDERATED_arg -> FEDERATED_arg[0](FEDERATED_arg[0](FEDERATED_arg[1])))'
    )

  def test_tf_wrapper_fails_bad_types(self):
    function = computation_types.FunctionType(
        None, computation_types.TensorType(tf.int32))
    federated = computation_types.FederatedType(tf.int32,
                                                placement_literals.CLIENTS)
    tuple_on_function = computation_types.NamedTupleType([federated, function])

    with self.assertRaisesRegexp(
        TypeError,
        r'you have attempted to create one with the type {int32}@CLIENTS'):

      @computation_wrapper_instances.tensorflow_wrapper(federated)
      def _(x):
        del x

    # pylint: disable=anomalous-backslash-in-string
    with self.assertRaisesRegexp(
        TypeError,
        r'you have attempted to create one with the type \( -> int32\)'):

      @computation_wrapper_instances.tensorflow_wrapper(function)
      def _(x):
        del x

    with self.assertRaisesRegexp(
        TypeError, r'you have attempted to create one with the type placement'):

      @computation_wrapper_instances.tensorflow_wrapper(
          computation_types.PlacementType())
      def _(x):
        del x

    with self.assertRaisesRegexp(
        TypeError, r'you have attempted to create one with the type T'):

      @computation_wrapper_instances.tensorflow_wrapper(
          computation_types.AbstractType('T'))
      def _(x):
        del x

    with self.assertRaisesRegexp(
        TypeError, r'you have attempted to create one with the type '
        '<{int32}@CLIENTS,\( -> int32\)>'):

      @computation_wrapper_instances.tensorflow_wrapper(tuple_on_function)
      def _(x):
        del x


class ToComputationImplTest(test.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      computation_wrapper_instances.building_block_to_computation(None)

  def test_converts_building_block_to_computation(self):
    lam = building_blocks.Lambda('x', tf.int32,
                                 building_blocks.Reference('x', tf.int32))
    computation_impl_lambda = computation_wrapper_instances.building_block_to_computation(
        lam)
    self.assertIsInstance(computation_impl_lambda,
                          computation_impl.ComputationImpl)

  def test_identity_lambda_executes_as_identity(self):
    lam = building_blocks.Lambda('x', tf.int32,
                                 building_blocks.Reference('x', tf.int32))
    computation_impl_lambda = computation_wrapper_instances.building_block_to_computation(
        lam)
    for k in range(10):
      self.assertEqual(computation_impl_lambda(k), k)


if __name__ == '__main__':
  test.main()

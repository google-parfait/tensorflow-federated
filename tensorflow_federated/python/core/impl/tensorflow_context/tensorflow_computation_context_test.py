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
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation_context
from tensorflow_federated.python.core.impl.types import placement_literals


class TensorFlowComputationContextTest(test_case.TestCase):

  def test_invoke_raises_value_error_with_federated_computation(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placement_literals.SERVER,
                                        True))
    def foo(x):
      return intrinsics.federated_broadcast(x)

    context = tensorflow_computation_context.TensorFlowComputationContext(
        tf.compat.v1.get_default_graph())

    with self.assertRaisesRegex(
        ValueError, 'Can only invoke TensorFlow in the body of '
        'a TensorFlow computation'):
      context.invoke(foo, None)

  def test_invoke_returns_result_with_tf_computation(self):
    make_10 = computations.tf_computation(lambda: tf.constant(10))
    add_one = computations.tf_computation(lambda x: tf.add(x, 1), tf.int32)

    @computations.tf_computation
    def add_one_with_v1(x):
      v1 = tf.Variable(1, name='v1')
      return x + v1

    @computations.tf_computation
    def add_one_with_v2(x):
      v2 = tf.Variable(1, name='v2')
      return x + v2

    @computations.tf_computation
    def foo():
      zero = tf.Variable(0, name='zero')
      ten = tf.Variable(make_10())
      return (add_one_with_v2(add_one_with_v1(add_one(make_10()))) + zero +
              ten - ten)

    graph = tf.compat.v1.Graph()
    context = tensorflow_computation_context.TensorFlowComputationContext(graph)

    self.assertEqual(foo.type_signature.compact_representation(), '( -> int32)')
    x = context.invoke(foo, None)

    with tf.compat.v1.Session(graph=graph) as sess:
      if context.init_ops:
        sess.run(context.init_ops)
      result = sess.run(x)
    self.assertEqual(result, 13)

if __name__ == '__main__':
  test_case.main()

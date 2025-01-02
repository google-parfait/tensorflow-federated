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

from absl.testing import absltest
import federated_language
from federated_language.proto import computation_pb2 as pb
import numpy as np
import tensorflow as tf
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation_context


class TensorFlowComputationContextTest(absltest.TestCase):

  def test_invoke_raises_value_error_with_federated_computation(self):
    bogus_proto = pb.Computation(
        type=federated_language.FunctionType(np.int32, np.int32).to_proto(),
        reference=pb.Reference(name='boogledy'),
    )
    non_tf_computation = federated_language.framework.ConcreteComputation(
        computation_proto=bogus_proto,
        context_stack=federated_language.framework.global_context_stack,
    )

    context = tensorflow_computation_context.TensorFlowComputationContext(
        tf.compat.v1.get_default_graph(), tf.constant('bogus_token')
    )

    with self.assertRaisesRegex(
        ValueError,
        'Can only invoke TensorFlow in the body of a TensorFlow computation',
    ):
      context.invoke(non_tf_computation, 1)

  def test_invoke_returns_result_with_tf_computation(self):
    make_10 = tensorflow_computation.tf_computation(lambda: tf.constant(10))
    add_one = tensorflow_computation.tf_computation(
        lambda x: tf.add(x, 1), tf.int32
    )

    @tensorflow_computation.tf_computation
    def add_one_with_v1(x):
      v1 = tf.Variable(1, name='v1')
      return x + v1

    @tensorflow_computation.tf_computation
    def add_one_with_v2(x):
      v2 = tf.Variable(1, name='v2')
      return x + v2

    @tensorflow_computation.tf_computation
    def foo():
      zero = tf.Variable(0, name='zero')
      ten = tf.Variable(make_10())
      return (
          add_one_with_v2(add_one_with_v1(add_one(make_10())))
          + zero
          + ten
          - ten
      )

    with tf.compat.v1.Graph().as_default() as graph:
      context = tensorflow_computation_context.TensorFlowComputationContext(
          graph, tf.constant('bogus_token')
      )

    self.assertEqual(foo.type_signature.compact_representation(), '( -> int32)')
    x = context.invoke(foo, None)

    with tf.compat.v1.Session(graph=graph) as sess:
      if context.init_ops:
        sess.run(context.init_ops)
      result = sess.run(x)
    self.assertEqual(result, 13)

  def test_get_session_token(self):
    @tensorflow_computation.tf_computation
    def get_the_token():
      return tensorflow_computation_context.get_session_token()

    with tf.compat.v1.Graph().as_default() as graph:
      context = tensorflow_computation_context.TensorFlowComputationContext(
          graph, tf.constant('test_token')
      )

    x = context.invoke(get_the_token, None)
    with tf.compat.v1.Session(graph=graph) as sess:
      result = sess.run(x)
    self.assertEqual(result, b'test_token')

  def test_get_session_token_nested(self):
    @tensorflow_computation.tf_computation
    def get_the_token_nested():
      return tensorflow_computation_context.get_session_token()

    @tensorflow_computation.tf_computation
    def get_the_token():
      return get_the_token_nested()

    with tf.compat.v1.Graph().as_default() as graph:
      context = tensorflow_computation_context.TensorFlowComputationContext(
          graph, tf.constant('test_token_nested')
      )

    x = context.invoke(get_the_token, None)
    with tf.compat.v1.Session(graph=graph) as sess:
      result = sess.run(x)
    self.assertEqual(result, b'test_token_nested')


if __name__ == '__main__':
  absltest.main()

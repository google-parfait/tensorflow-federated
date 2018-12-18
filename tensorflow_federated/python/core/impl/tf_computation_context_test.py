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
"""Tests for tf_computation_context.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import tf_computation_context
from tensorflow_federated.python.core.impl.context_stack import context_stack


class TensorFlowComputationContextTest(absltest.TestCase):

  def test_invoke_federated_computation_fails(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.SERVER, True))
    def foo(x):
      return intrinsics.federated_broadcast(x)

    context = tf_computation_context.TensorFlowComputationContext(
        tf.get_default_graph())
    with self.assertRaisesRegexp(ValueError,
                                 'Expected a TensorFlow computation.'):
      context.invoke(foo, None)

  def test_invoke_tf_computation(self):
    make_10 = computations.tf_computation(lambda: tf.constant(10))
    add_one = computations.tf_computation(lambda x: tf.add(x, 1), tf.int32)

    @computations.tf_computation
    def foo():
      return add_one(add_one(add_one(make_10())))

    self.assertEqual(str(foo.type_signature), '( -> int32)')
    with tf.Graph().as_default() as graph:
      context = tf_computation_context.TensorFlowComputationContext(
          tf.get_default_graph())
      with context_stack.install(context):
        with tf.Session(graph=graph) as sess:
          self.assertEqual(sess.run(foo()), 13)


if __name__ == '__main__':
  absltest.main()

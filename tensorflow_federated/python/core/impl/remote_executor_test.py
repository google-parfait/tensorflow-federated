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
"""Tests for remote_executor.py."""

import collections
import contextlib

from absl.testing import absltest
import grpc
from grpc.framework.foundation import logging_pool
import portpicker
import tensorflow as tf

from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import eager_executor
from tensorflow_federated.python.core.impl import executor_service
from tensorflow_federated.python.core.impl import executor_test_utils
from tensorflow_federated.python.core.impl import lambda_executor
from tensorflow_federated.python.core.impl import remote_executor
from tensorflow_federated.python.core.impl import set_default_executor


@contextlib.contextmanager
def test_context():
  port = portpicker.pick_unused_port()
  server_pool = logging_pool.pool(max_workers=1)
  server = grpc.server(server_pool)
  server.add_insecure_port('[::]:{}'.format(port))
  target_executor = eager_executor.EagerExecutor()
  tracer = executor_test_utils.TracingExecutor(target_executor)
  service = executor_service.ExecutorService(tracer)
  executor_pb2_grpc.add_ExecutorServicer_to_server(service, server)
  server.start()
  channel = grpc.insecure_channel('localhost:{}'.format(port))
  executor = lambda_executor.LambdaExecutor(
      remote_executor.RemoteExecutor(channel))
  set_default_executor.set_default_executor(executor)
  yield collections.namedtuple('_', 'executor tracer')(executor, tracer)
  set_default_executor.set_default_executor()
  try:
    channel.close()
  except AttributeError:
    del channel
  server.stop(None)


class RemoteExecutorTest(absltest.TestCase):

  def test_no_arg_tf_computation(self):
    with test_context():

      @computations.tf_computation
      def comp():
        return 10

      self.assertEqual(comp(), 10)

  def test_one_arg_tf_computation(self):
    with test_context():

      @computations.tf_computation(tf.int32)
      def comp(x):
        return x + 1

      self.assertEqual(comp(10), 11)

  def test_two_arg_tf_computation(self):
    with test_context():

      @computations.tf_computation(tf.int32, tf.int32)
      def comp(x, y):
        return x + y

      self.assertEqual(comp(10, 20), 30)

  def test_with_selection(self):
    with test_context() as context:

      @computations.tf_computation(tf.int32)
      def foo(x):
        return collections.OrderedDict([('A', x + 10), ('B', x + 20)])

      @computations.tf_computation(tf.int32, tf.int32)
      def bar(x, y):
        return x + y

      @computations.federated_computation(tf.int32)
      def baz(x):
        return bar(foo(x).A, foo(x).B)

      self.assertEqual(baz(100), 230)

      # Make sure exactly two selections happened.
      self.assertLen(
          [x for x in context.tracer.trace if x[0] == 'create_selection'], 2)

  def test_with_mnist_training_example(self):
    with test_context() as context:
      executor_test_utils.test_mnist_training(self, context.executor)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()

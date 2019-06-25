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
"""Tests for executor_service.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import grpc
from grpc.framework.foundation import logging_pool
import portpicker

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.core.impl import executor_service


class ExecutorServiceTest(absltest.TestCase):

  def setUp(self):
    super(ExecutorServiceTest, self).setUp()
    port = portpicker.pick_unused_port()
    server_pool = logging_pool.pool(max_workers=1)
    self._server = grpc.server(server_pool)
    self._server.add_insecure_port('[::]:{}'.format(port))
    service = executor_service.ExecutorService()
    executor_pb2_grpc.add_ExecutorServicer_to_server(service, self._server)
    self._server.start()
    self._channel = grpc.insecure_channel('localhost:%d' % port)
    self._stub = executor_pb2_grpc.ExecutorStub(self._channel)

  def tearDown(self):
    # TODO(b/134543154): Find some way of cleanly disposing of channels that is
    # consistent between Google-internal and OSS stacks.
    try:
      self._channel.close()
    except AttributeError:
      # The `.close()` method does not appear to be present in grpcio 1.8.6, so
      # we have to fall back on explicitly calling the destructor.
      del self._stub
      del self._channel
    self._server.stop(None)
    super(ExecutorServiceTest, self).tearDown()

  def test_executor_service_constructor_with_no_args(self):
    value = executor_pb2.Value()
    request = executor_pb2.CreateValueRequest(value=value)
    response = self._stub.CreateValue(request)
    self.assertIsInstance(response, executor_pb2.CreateValueResponse)


if __name__ == '__main__':
  absltest.main()

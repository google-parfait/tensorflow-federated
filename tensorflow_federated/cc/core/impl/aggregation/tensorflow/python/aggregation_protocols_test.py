# Copyright 2022 Google LLC
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
"""Tests for aggregation_protocols."""

import tempfile
from typing import Any

from absl.testing import absltest
import tensorflow as tf

from pybind11_abseil import status  # pylint:disable=unused-import
from tensorflow_federated.cc.core.impl.aggregation.core import tensor_pb2
from tensorflow_federated.cc.core.impl.aggregation.protocol import aggregation_protocol_messages_pb2 as apm_pb2
from tensorflow_federated.cc.core.impl.aggregation.protocol import configuration_pb2
from tensorflow_federated.cc.core.impl.aggregation.protocol.python import aggregation_protocol  # pylint:disable=unused-import
from tensorflow_federated.cc.core.impl.aggregation.tensorflow.python import aggregation_protocols


def create_client_input(tensors: dict[str, Any]) -> apm_pb2.ClientMessage:
  with tempfile.NamedTemporaryFile() as tmpfile:
    tf.raw_ops.Save(
        filename=tmpfile.name,
        tensor_names=list(tensors.keys()),
        data=list(tensors.values()))
    with open(tmpfile.name, 'rb') as f:
      return apm_pb2.ClientMessage(
          simple_aggregation=apm_pb2.ClientMessage.SimpleAggregation(
              input=apm_pb2.ClientResource(inline_bytes=f.read())))


class AggregationProtocolsTest(absltest.TestCase):

  def test_simple_aggregation_protocol(self):
    input_tensor = tensor_pb2.TensorSpecProto(
        name='in', dtype=tensor_pb2.DT_INT32, shape={}
    )
    output_tensor = tensor_pb2.TensorSpecProto(
        name='out', dtype=tensor_pb2.DT_INT32, shape={}
    )
    config = configuration_pb2.Configuration(
        intrinsic_configs=[
            configuration_pb2.Configuration.IntrinsicConfig(
                intrinsic_uri='federated_sum',
                intrinsic_args=[
                    configuration_pb2.Configuration.IntrinsicConfig.IntrinsicArg(
                        input_tensor=input_tensor
                    ),
                ],
                output_tensors=[output_tensor],
            ),
        ]
    )

    agg_protocol = aggregation_protocols.create_simple_aggregation_protocol(
        config
    )
    self.assertIsNotNone(agg_protocol)

    agg_protocol.Start(2)
    start_client_id = 0

    agg_protocol.ReceiveClientMessage(
        start_client_id, create_client_input({input_tensor.name: 3}))
    agg_protocol.ReceiveClientMessage(
        start_client_id + 1, create_client_input({input_tensor.name: 5}))

    agg_protocol.Complete()
    with tempfile.NamedTemporaryFile('wb') as tmpfile:
      tmpfile.write(agg_protocol.GetResult())
      tmpfile.flush()
      self.assertEqual(
          tf.raw_ops.Restore(
              file_pattern=tmpfile.name,
              tensor_name=output_tensor.name,
              dt=output_tensor.dtype), 8)


if __name__ == '__main__':
  absltest.main()

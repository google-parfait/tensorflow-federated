# Copyright 2022, The TensorFlow Federated Authors.
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
"""An example of running a TFF computation with a scalar-value DataBackend."""

from typing import Sequence

from absl import app
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.proto.v0 import computation_pb2 as pb


# DataBackend, which acts as an interface to stored data, returning a scalar
# value.
class ScalarDataBackend(tff.framework.DataBackend):

  async def materialize(self, data, type_spec):
    return tf.cast(int(data.uri[-1]), tf.int32)


def main(_: Sequence[str]) -> None:

  def ex_fn(device: tf.config.LogicalDevice) -> tff.framework.DataExecutor:
    # In order to de-reference data uri's bundled in TFF computations, a
    # DataExecutor must exist in the runtime context to process those uri's and
    # return the underlying data. We can wrap an EagerTFExecutor (which handles
    # TF operations) with a DataExecutor instance defined with a DataBackend
    # object.
    return tff.framework.DataExecutor(
        tff.framework.EagerTFExecutor(device), data_backend=ScalarDataBackend())

  # Executor factory used by the runtime context to spawn executors to run TFF
  # computations.
  factory = tff.framework.local_executor_factory(leaf_executor_fn=ex_fn)

  # Context in which to execute the following computation.
  ctx = tff.framework.ExecutionContext(executor_fn=factory)
  tff.framework.set_default_context(ctx)

  # Type of the data returned by the DataBackend.
  element_type = tff.types.TensorType(tf.int32)
  element_type_proto = tff.framework.serialize_type(element_type)
  # We construct a list of uri's as our references to the dataset.
  uris = [f'uri://{i}' for i in range(3)]
  # The uris are embedded in TFF computation protos so they can be processed by
  # TFF executors.
  arguments = [
      pb.Computation(data=pb.Data(uri=uri), type=element_type_proto)
      for uri in uris
  ]
  # The embedded uris are passed to a DataDescriptor which recognizes the
  # underlying dataset as federated and allows combining it with a federated
  # computation.
  data_handle = tff.framework.DataDescriptor(
      None, arguments, tff.FederatedType(element_type, tff.CLIENTS),
      len(arguments))

  # Federated computation that sums the scalars.
  @tff.federated_computation(tff.types.FederatedType(tf.int32, tff.CLIENTS))
  def foo(x):
    return tff.federated_sum(x)

  # Should print 3.
  print(foo(data_handle))


if __name__ == '__main__':
  app.run(main)

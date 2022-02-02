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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""Python interface to C++ Executor implementations."""

import asyncio

import tensorflow_federated as tff

from tensorflow_federated.examples.custom_data_backend import data_backend_example_bindings
from tensorflow_federated.proto.v0 import computation_pb2 as tff_proto


class DataBackendExample(tff.framework.DataBackend):
  """An example DataBackend implemented in C++."""

  def __init__(self):
    self.cc_ = data_backend_example_bindings.DataBackendExample()

  async def materialize(self, data: tff_proto.Data, type_spec: tff.Type):
    """Materializes `data` with the given `type_spec`.

    Args:
      data: A symbolic reference to the data to be materialized locally. Must be
        an instance of `pb.Data`.
      type_spec: An instance of `computation_types.Type` that represents the
        type of the data payload being materialized.

    Returns:
      The materialized payload.
    """

    def blocking_materialize():
      type_spec_proto = tff.framework.serialize_type(type_spec)
      proto = self.cc_.resolve_to_value(data, type_spec_proto)
      value, _ = tff.framework.deserialize_value(proto)
      return value

    # Run the blocking method on a separate thread from the event loop
    return await asyncio.get_running_loop().run_in_executor(
        None, blocking_materialize)

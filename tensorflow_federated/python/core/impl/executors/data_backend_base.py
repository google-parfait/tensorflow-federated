# Copyright 2021, The TensorFlow Federated Authors.
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
"""Abstract interface for data backends."""

import abc
from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.impl.types import computation_types


class DataBackend(metaclass=abc.ABCMeta):
  """Abstract interface for data backends.

  A data backend is a component that can resolve symbolic references to data
  as URIs, and locally materialize the associated payloads. Data backends are
  used in tandem with the `data_executor` that queries them as it encounters
  the `data` building block.
  """

  @abc.abstractmethod
  async def materialize(self, data: pb.Data, type_spec: computation_types.Type):
    """Materializes `data` with the given `type_spec`.

    The form of the materialized payload must be such that it can be understood
    by the downstream components of the executor stack. For example, if the data
    backend is plugged into a stack based on an eager TensorFlow executor, the
    accepted forms of payload would include Numpy-like objects, tensors, as well
    as instances of eager `tf.data.Dataset`, and structures thereof. It is the
    responsibility of the code that constructs the executor stack with the given
    data backend to ensure that the types of payload materialized are compatible
    with what the downstream components of the executor stack can accept.

    Args:
      data: A symbolic reference to the data to be materialized locally. Must be
        an instance of `pb.Data`.
      type_spec: An instance of `computation_types.Type` that represents the
        type of the data payload being materialized.

    Returns:
      The materialized payload.
    """
    raise NotImplementedError

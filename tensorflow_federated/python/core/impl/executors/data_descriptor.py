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
"""Helper class for representing fully-specified data-yeilding computations."""

import asyncio
from collections.abc import Mapping
from typing import Any, Optional

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.executors import cardinality_carrying_base
from tensorflow_federated.python.core.impl.executors import ingestable_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types.type_serialization import serialize_type


def CreateDataDescriptor(arg_uris: list[str], arg_type: computation_types.Type):
  """Constructs a `DataDescriptor` instance targeting a `tff.DataBackend`.

  Args:
    arg_uris: List of URIs compatible with the data backend embedded in the
      given `tff.framework.SyncExecutionContext`.
    arg_type: The type of data referenced by the URIs. An instance of
      `tff.Type`.

  Returns:
    Instance of `DataDescriptor`
  """
  arg_type_proto = serialize_type(arg_type)
  args = [
      pb.Computation(data=pb.Data(uri=uri), type=arg_type_proto)
      for uri in arg_uris
  ]
  return DataDescriptor(
      None,
      args,
      computation_types.FederatedType(arg_type, placements.CLIENTS),
      len(args),
  )


class CardinalityFreeDataDescriptor(ingestable_base.Ingestable):
  """Represent data-yielding computations with unspecified cardinalities.

  Instances of this class are objects that may combine a federated computation
  that returns federated data and an argument to be supplied as input to this
  computation, or alternatively the argument alone. These objects are designed
  to be recognized by the runtime. When ingesting those objects (e.g., as they
  are passed as arguments to a computation invocation), the runtime ingests
  the argument, and (if provided) invokes the computation contained
  in this descriptor on this argument to cause the data to materialize within
  the runtime (but without marshaling it out and returning it to user).

  In the typical usage of this helper class, the embedded argument is a set of
  handles, and the embedded computation transforms those handles into physical
  `tf.data.Dataset` instances. This transformation occurs on the clients (i.e.,
  in the TFF runtime worker processes).

  Alternatively, the argument itself may consist of computations to be locally
  executed on the clients. In this case, the computation can be omitted.
  """

  def __init__(
      self,
      comp: Optional[computation_base.Computation],
      arg: Any,
      arg_type: computation_types.Type,
  ):
    """Constructs this data descriptor from the given computation and argument.

    Args:
      comp: The computation that materializes the data, of some type `(T -> U)`
        where `T` is the type of the argument `arg` and `U` is the type of the
        materialized data that's being produced. This can be `None`, in which
        case it's assumed to be an identity function (and `T` in that case must
        be identical to `U`).
      arg: The argument to be passed as input to `comp` if `comp` is not `None`,
        or to be treated as the computed result. Must be recognized by the TFF
        runtime as a payload of type `T`.
      arg_type: The type of the argument (`T` references above). An instance of
        `tff.Type`.

    Raises:
      ValueError: if the arguments don't satisfy the constraints listed above.
    """
    self._comp = comp
    self._arg = arg
    self._arg_type = computation_types.to_type(arg_type)
    if self._comp is not None:
      if not self._comp.type_signature.parameter.is_assignable_from(
          self._arg_type
      ):
        raise ValueError(
            'Argument type {} incompatible with the computation '
            'parameter {}.'.format(
                str(self._arg_type), str(self._comp.type_signature.parameter)
            )
        )
      self._type_signature = self._comp.type_signature.result
    else:
      self._type_signature = self._arg_type

  @property
  def type_signature(self):
    return self._type_signature

  async def ingest(self, executor):
    if isinstance(self._arg, ingestable_base.Ingestable):
      arg_coro = self._arg.ingest(executor)
    else:
      if self._comp is not None:
        expected_arg_type = self._comp.type_signature.parameter
      else:
        expected_arg_type = self._arg_type
      arg_coro = executor.create_value(self._arg, expected_arg_type)
    if self._comp is not None:
      comp_val, arg_val = await asyncio.gather(
          executor.create_value(self._comp, self._comp.type_signature), arg_coro
      )
      return await executor.create_call(comp_val, arg_val)
    else:
      return await arg_coro


class DataDescriptor(
    CardinalityFreeDataDescriptor, cardinality_carrying_base.CardinalityCarrying
):
  """Represents fully-specified data-yielding computations.

  Similar to CardinalityFreeDataDescriptor, but additionally accepts a
  cardinality argument, allowing callers to explcitly specify the number
  of clients this data descriptor is intended to represent.
  """

  def __init__(
      self,
      comp: Optional[computation_base.Computation],
      arg: Any,
      arg_type: computation_types.Type,
      cardinality: Optional[int] = None,
  ):
    """Constructs this data descriptor from the given computation and argument.

    Args:
      comp: The computation that materializes the data, of some type `(T -> U)`
        where `T` is the type of the argument `arg` and `U` is the type of the
        materialized data that's being produced. This can be `None`, in which
        case it's assumed to be an identity function (and `T` in that case must
        be identical to `U`).
      arg: The argument to be passed as input to `comp` if `comp` is not `None`,
        or to be treated as the computed result. Must be recognized by the TFF
        runtime as a payload of type `T`.
      arg_type: The type of the argument (`T` references above). An instance of
        `tff.Type`.
      cardinality: If of federated type, placed at clients, this int specifies
        the number of clients represented by this DataDescriptor.

    Raises:
      ValueError: if the arguments don't satisfy the constraints listed above.
    """
    super().__init__(comp, arg, arg_type)
    self._cardinality: dict[placements.PlacementLiteral, int] = {}
    if self._type_signature.is_federated():
      if self._type_signature.placement is placements.CLIENTS:
        if cardinality is None:
          raise ValueError('Expected `cardinality` to not be `None`.')
        self._cardinality[placements.CLIENTS] = cardinality
      else:
        if cardinality is not None:
          raise ValueError(
              f'Expected `cardinality` to be `None`, found: {cardinality}.'
          )

  @property
  def cardinality(self) -> Mapping[placements.PlacementLiteral, int]:
    return self._cardinality

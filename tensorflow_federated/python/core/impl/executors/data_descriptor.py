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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""Helper class for representing fully-specified data-yeilding computations."""

import asyncio
from typing import Any, Optional

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.impl.executors import cardinality_carrying_base
from tensorflow_federated.python.core.impl.executors import ingestable_base
from tensorflow_federated.python.core.impl.types import computation_types


class DataDescriptor(ingestable_base.Ingestable,
                     cardinality_carrying_base.CardinalityCarrying):
  """Helper class for representing fully-specified data-yielding computations.

  Instances of this class are objects that may combine a federated computation
  that returns a portion of federated data and an argument to be supplied as
  input to this computation, or the argument alone. These objects are designed
  to be recognized by the runtime. When ingesting those objects (e.g., as they
  are passed as arguments to a computation invocation), the runtime ingests
  the argument, and (if provided) optionally invokes the computation contained
  in this descriptor on the argument to cause the data to materialize within
  the runtime (but without marshaling it out and returning it to user).

  In the typical usage of this helper class, the embedded argument is a set of
  handles, and the embedded computation transforms those handles into physical
  `tf.data.Dataset` instances. This transformation occurs on the clients (i.e.,
  in the TFF runtime worker processes).

  Alternatively, the argument itself may consist of computations to be locally
  executed on the clients. In this case, the computation can be omitted.
  """

  def __init__(self,
               comp: Optional[computation_base.Computation],
               arg: Any,
               arg_type: computation_types.Type,
               cardinality: Optional[int] = None):
    """Constructs this data descriptor from the given computation and argument.

    Args:
      comp: The computation that materializes the data, of some type `(T -> U)`
        where `T` is the type of the argument `arg` and `U` is the type of the
        materialzied data that's being produces. This can be `None`, in which
        case it's assumed to be an identity function (and `T` in that case must
        be identical to `U`).
      arg: The argument to be passed as input to `comp` if `comp` is not `None`,
        or to be treated as the computed result. Must be recognized by the TFF
        runtime as a payload of type `T`.
      arg_type: The type of the argument (`T` references above). An instance of
        `tff.Type`.
      cardinality: Optional cardinality (e.g., number of clients) to supply if
        this descriptor represents an instance of federated data.

    Raises:
      ValueError: if the arguments don't satisfy the constraints listed above.
    """
    self._comp = comp
    self._arg = arg
    self._arg_type = arg_type
    if self._comp is not None:
      if not self._comp.type_signature.parameter.is_assignable_from(
          self._arg_type):
        raise ValueError('Argument type {} incompatible with the computation '
                         'parameter {}.'.format(
                             str(self._arg_type),
                             str(self._comp.type_signature.parameter)))
      self._type_signature = self._comp.type_signature.result
    else:
      self._type_signature = self._arg_type
    self._cardinality = {}
    if (isinstance(self._type_signature, computation_types.FederatedType) and
        cardinality is not None):
      self._cardinality[self._type_signature.placement] = cardinality

  @property
  def type_signature(self):
    return self._type_signature

  @property
  def cardinality(self):
    return self._cardinality

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
          executor.create_value(self._comp, self._comp.type_signature),
          arg_coro)
      return await executor.create_call(comp_val, arg_val)
    else:
      return await arg_coro

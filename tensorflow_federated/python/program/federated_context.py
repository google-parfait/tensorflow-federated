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
"""Defines an abstract interface for representing a federated context."""

import abc
from typing import Optional, Union

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import get_context_stack
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.program import structure_utils
from tensorflow_federated.python.program import value_reference


ComputationArgValue = Union[
    value_reference.MaterializableStructure,
    object,
    computation_base.Computation,
]


def contains_only_server_placed_data(
    type_signature: computation_types.Type,
) -> bool:
  """Determines if `type_signature` contains only server-placed data.

  Determines if `type_signature` contains only:
  * `tff.StructType`s
  * `tff.SequenceType`s
  * server-placed `tff.FederatedType`s
  * `tff.TensorType`s

  Args:
      type_signature: The `tff.Type` to test.

  Returns:
    `True` if `type_signature` contains only server-placed data, otherwise
    `False`.
  """
  py_typecheck.check_type(type_signature, computation_types.Type)

  def predicate(type_spec: computation_types.Type) -> bool:
    return isinstance(
        type_spec,
        (
            computation_types.StructType,
            computation_types.SequenceType,
            computation_types.TensorType,
        ),
    ) or (
        isinstance(type_spec, computation_types.FederatedType)
        and type_spec.placement is placements.SERVER
    )

  return type_analysis.contains_only(type_signature, predicate)


class FederatedContext(context_base.SyncContext):
  """An abstract interface representing a federated context.

  A federated context supports invoking a limited set of `tff.Computation`s,
  making guarantees about what a `tff.Computation` can accept as an argument and
  what it returns when invoked.

  ## Restrictions on the TensorFlow Federated Type

  Arguments can be nested structures of values corresponding to the TensorFlow
  Federated type signature of the `tff.Computation`:

  *   Server-placed values must be represented by
      `tff.program.MaterializableStructure`.
  *   Client-placed values must be represented by structures of values returned
      by a `tff.program.FederatedDataSourceIterator`.

  Return values can be structures of `tff.program.MaterializableValueReference`s
  or a single `tff.program.MaterializableValueReference`, where a reference
  corresponds to the tensor-type of the TensorFlow Federated type signature in
  the return value of the invoked `tff.Computation`.

  ## TensorFlow Federated Type to Python Representation

  In order to interact with the value returned by a `tff.Computation`, it is
  helpful to be able to reason about the Python type of this value. In some way
  this Python type must depend on the TensorFlow Federated type signature of the
  associated value. To provide uniformity of experience and ease of reasoning,
  we specify the Python representation of values in a manner that can be stated
  entirely in the TensorFlow Federated typesystem.

  We have chosen to limit the TensorFlow Federated type signatures of invoked
  `tff.Computation`s to disallow the returning of client-placed values,
  `tff.SequenceTypes`, and `tff.FunctionTypes`, in order to reduced the area
  which needs to be supported by federated programs. Below we describe the
  mapping between TensorFlow Federated type signatures and Python
  representations of values that can be passed as arguments to or returned as
  results from `tff.Computation`s.

  Python representations of values that can be *accepted as an arguments to* or
  *returned as a value from* a `tff.Computation`:

  | TensorFlow Federated Type  | Python Representation                      |
  | -------------------------- | ------------------------------------------ |
  | `tff.TensorType`           | `tff.program.MaterializableValueReference` |
  | `tff.SequenceType`         | `tff.program.MaterializableValueReference` |
  | `tff.FederatedType`        | Python representation of the `member` of   |
  : (server-placed)            : the `tff.FederatedType`                    :
  | `tff.StructWithPythonType` | Python container of the                    |
  :                            : `tff.StructWithPythonType`                 :
  | `tff.StructType` (with no  | `collections.OrderedDict`                  |
  : Python type, all fields    :                                            :
  : named)                     :                                            :
  | `tff.StructType` (with no  | `tuple`                                    |
  : Python type, no fields     :                                            :
  : named)                     :                                            :

  Python representations of values that can be only be *accepted as an arguments
  to* a `tff.Computation`:

  | TFF Type            | Python Representation                   |
  | ------------------- | --------------------------------------- |
  | `tff.FederatedType` | Opaque object returned by               |
  : (client-placed)     : `tff.program.DataSourceIterator.select` :
  | `tff.FunctionType`  | `tff.Computation`                       |
  """

  @abc.abstractmethod
  def invoke(
      self,
      comp: computation_base.Computation,
      arg: Optional[ComputationArgValue],
  ) -> structure_utils.Structure[value_reference.MaterializableValueReference]:
    """Invokes the `comp` with the argument `arg`.

    Args:
      comp: The `tff.Computation` being invoked.
      arg: The optional argument of `comp`; server-placed values must be
        represented by `tff.program.MaterializableStructure`, and client-placed
        values must be represented by structures of values returned by a
        `tff.program.FederatedDataSourceIterator`.

    Returns:
      The result of invocation; a structure of
      `tff.program.MaterializableValueReference`.

    Raises:
      ValueError: If the result type of `comp` does not contain only structures,
      server-placed values, or tensors.
    """
    raise NotImplementedError


def check_in_federated_context() -> None:
  """Checks if the current context is a `tff.program.FederatedContext`."""
  context_stack = get_context_stack.get_context_stack()
  if not isinstance(context_stack.current, FederatedContext):
    raise ValueError(
        'Expected the current context to be a `tff.program.FederatedContext`, '
        f'found {type(context_stack.current)}.'
    )

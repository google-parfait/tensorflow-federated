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
"""A compiler for the xla backend."""

from jax.lib import xla_client
import numpy as np

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.xla_backend import xla_serialization
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis


class XlaComputationFactory:
  """An implementation of local computation factory for XLA computations."""

  def __init__(self):
    pass

  def create_constant_from_scalar(
      self, value, type_spec: computation_types.Type
  ) -> tuple[computation_pb2.Computation, computation_types.Type]:
    """Creates a TFF computation returning a constant based on a scalar value.

    The returned computation has the type signature `( -> T)`, where `T` may be
    either a scalar, or a nested structure made up of scalars.

    Args:
      value: A numpy scalar representing the value to return from the
        constructed computation (or to broadcast to all parts of a nested
        structure if `type_spec` is a structured type).
      type_spec: A `computation_types.Type` of the constructed constant. Must be
        either a tensor, or a nested structure of tensors.

    Returns:
      A tuple `(pb.Computation, computation_types.Type)` with the first element
      being a TFF computation with semantics as described above, and the second
      element representing the formal type of that computation.
    """
    py_typecheck.check_type(type_spec, computation_types.Type)
    if not type_analysis.is_structure_of_tensors(type_spec):
      raise ValueError(
          'Not a tensor or a structure of tensors: {}'.format(str(type_spec))
      )

    builder = xla_client.XlaBuilder('comp')

    def _constant_from_tensor(tensor_type):
      py_typecheck.check_type(tensor_type, computation_types.TensorType)
      numpy_value = np.full(
          shape=tensor_type.shape,
          fill_value=value,
          dtype=tensor_type.dtype,
      )
      return xla_client.ops.Constant(builder, numpy_value)

    if isinstance(type_spec, computation_types.TensorType):
      tensors = [_constant_from_tensor(type_spec)]
    else:
      tensors = [_constant_from_tensor(x) for x in structure.flatten(type_spec)]

    # Likewise, results are always returned as a single tuple with results.
    # This is always a flat tuple; the nested TFF structure is defined by the
    # binding.
    xla_client.ops.Tuple(builder, tensors)
    xla_computation = builder.build()

    comp_type = computation_types.FunctionType(None, type_spec)
    comp_pb = xla_serialization.create_xla_tff_computation(
        xla_computation, [], comp_type
    )
    return (comp_pb, comp_type)

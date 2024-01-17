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

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.xla_backend import xla_serialization
from tensorflow_federated.python.core.impl.compiler import local_computation_factory_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis


class XlaComputationFactory(
    local_computation_factory_base.LocalComputationFactory
):
  """An implementation of local computation factory for XLA computations."""

  def __init__(self):
    pass

  def create_constant_from_scalar(
      self, value, type_spec: computation_types.Type
  ) -> local_computation_factory_base.ComputationProtoAndType:
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

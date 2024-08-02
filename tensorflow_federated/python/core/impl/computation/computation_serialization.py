# Copyright 2018, The TensorFlow Federated Authors.
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
"""Utilities for serializing and deserializing TFF computations."""

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl


def serialize_computation(
    computation: computation_base.Computation,
) -> pb.Computation:
  """Serializes 'tff.Computation' as a pb.Computation.

  Note: Currently only serialization for computation_impl.ConcreteComputation is
  implemented.

  Args:
    computation: An instance of `tff.Computation`.

  Returns:
    The corresponding instance of `pb.Computation`.

  Raises:
    TypeError: If the argument is of the wrong type.
    NotImplementedError: for computation variants for which serialization is not
      implemented.
  """
  py_typecheck.check_type(computation, computation_base.Computation)

  if isinstance(computation, computation_impl.ConcreteComputation):
    computation_proto = pb.Computation()
    computation_proto.CopyFrom(
        computation_impl.ConcreteComputation.get_proto(computation)
    )
    return computation_proto
  else:
    raise NotImplementedError(
        'Serialization of type {} is not currentlyimplemented yet.'.format(
            type(computation)
        )
    )


def deserialize_computation(
    computation_proto: pb.Computation,
) -> computation_base.Computation:
  """Deserializes 'tff.Computation' as a pb.Computation.

  Args:
    computation_proto: An instance of `pb.Computation`.

  Returns:
    The corresponding instance of `tff.Computation`.

  Raises:
    TypeError: If the argument is of the wrong type.
  """
  py_typecheck.check_type(computation_proto, pb.Computation)
  return computation_impl.ConcreteComputation(
      computation_proto=computation_proto,
      context_stack=context_stack_impl.context_stack,
  )

# Copyright 2020, The TensorFlow Federated Authors.
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
"""A library of contruction functions for computation structures."""

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.types import type_factory
from tensorflow_federated.python.core.impl.types import type_serialization


def create_lambda_empty_tuple() -> pb.Computation:
  """Returns a lambda computation returning an empty tuple.

  Has the type signature:

  ( -> <>)

  Returns:
    An instance of `pb.Computation`.
  """
  result_type = computation_types.NamedTupleType([])
  type_signature = computation_types.FunctionType(None, result_type)
  result = pb.Computation(
      type=type_serialization.serialize_type(result_type),
      tuple=pb.Tuple(element=[]))
  fn = pb.Lambda(parameter_name=None, result=result)
  # We are unpacking the lambda argument here because `lambda` is a reserved
  # keyword in Python, but it is also the name of the parameter for a
  # `pb.Computation`.
  # https://developers.google.com/protocol-buffers/docs/reference/python-generated#keyword-conflicts
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature), **{'lambda': fn})  # pytype: disable=wrong-keyword-args


def create_lambda_identity(type_spec) -> pb.Computation:
  """Returns a lambda computation representing an identity function.

  Has the type signature:

  (T -> T)

  Args:
    type_spec: A type convertible to instance of `computation_types.Type` via
      `computation_types.to_type`.

  Returns:
    An instance of `pb.Computation`.
  """
  type_spec = computation_types.to_type(type_spec)
  type_signature = type_factory.unary_op(type_spec)
  result = pb.Computation(
      type=type_serialization.serialize_type(type_spec),
      reference=pb.Reference(name='a'))
  fn = pb.Lambda(parameter_name='a', result=result)
  # We are unpacking the lambda argument here because `lambda` is a reserved
  # keyword in Python, but it is also the name of the parameter for a
  # `pb.Computation`.
  # https://developers.google.com/protocol-buffers/docs/reference/python-generated#keyword-conflicts
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature), **{'lambda': fn})  # pytype: disable=wrong-keyword-args

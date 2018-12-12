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
"""Defines the implementation of the base Computation interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.proto.v0 import computation_pb2 as pb

from tensorflow_federated.python.common_libs import py_typecheck

from tensorflow_federated.python.core.impl import func_utils
from tensorflow_federated.python.core.impl import type_serialization
from tensorflow_federated.python.core.impl import type_utils

from tensorflow_federated.python.core.impl.context_stack import context_stack


class ComputationImpl(func_utils.ConcreteFunction):
  """An implementation of the base interface cb.Computation."""

  @classmethod
  def get_proto(cls, value):
    py_typecheck.check_type(value, cls)
    return value._computation_proto  # pylint: disable=protected-access

  def __init__(self, computation_proto):
    """Constructs a new instance of ComputationImpl from the computation_proto.

    Args:
      computation_proto: The protocol buffer that represents the computation, an
        instance of pb.Computation.
    """
    py_typecheck.check_type(computation_proto, pb.Computation)
    type_spec = type_serialization.deserialize_type(computation_proto.type)
    type_utils.check_well_formed(type_spec)
    super(ComputationImpl, self).__init__(type_spec)
    self._computation_proto = computation_proto

  def _invoke(self, arg):
    return context_stack.current.invoke(self, arg)

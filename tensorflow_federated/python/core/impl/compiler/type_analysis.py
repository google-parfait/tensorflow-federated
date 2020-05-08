# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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
"""A library of static analysis functions for computation types."""

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import type_transformations


def _visit_type(type_signature, function):

  def inner_function(inner_type):
    function(inner_type)
    return inner_type, False

  type_transformations.transform_type_postorder(type_signature, inner_function)


def _count(type_signature, predicate):
  found = 0

  def add_if(inner_type):
    nonlocal found
    found += 1 if predicate(inner_type) else 0

  _visit_type(type_signature, add_if)
  return found


def contains_federated_type(type_signature):
  """Returns whether or not `type_signature` contains a federated type."""
  return _count(type_signature,
                lambda t: isinstance(t, computation_types.FederatedType)) > 0


def contains_tensors(type_signature):
  """Returns whether or not `type_signature` contains a tensor type."""
  return _count(type_signature,
                lambda t: isinstance(t, computation_types.TensorType)) > 0

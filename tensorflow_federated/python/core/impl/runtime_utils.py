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
"""Common functions needed across executor classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import six

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import placement_literals


def merge_cardinalities(existing, to_add):
  """Merges dicts `existing` and `to_add`, checking for conflicts."""
  py_typecheck.check_type(existing, dict)
  py_typecheck.check_type(to_add, dict)
  for key, val in six.iteritems(existing):
    py_typecheck.check_type(key, placement_literals.PlacementLiteral)
    py_typecheck.check_type(val, int)
  if not to_add:
    return existing
  elif not existing:
    return to_add
  cardinalities = {}
  cardinalities.update(existing)
  for key, val in six.iteritems(to_add):
    py_typecheck.check_type(key, placement_literals.PlacementLiteral)
    py_typecheck.check_type(val, int)
    if key not in cardinalities:
      cardinalities[key] = val
    elif cardinalities[key] != val:
      raise ValueError('Conflicting cardinalities for {}: {} vs {}'.format(
          key, val, cardinalities[key]))
  return cardinalities


def infer_cardinalities(value, type_spec):
  """Infers cardinalities from Python `value`.

  Allows for any Python object to represent a federated value; enforcing
  particular representations is not the job of this inference function, but
  rather ingestion functions lower in the stack.

  Args:
    value: Python object from which to infer TFF placement cardinalities.
    type_spec: The TFF type spec for `value`, determining the semantics for
      inferring cardinalities. That is, we only pull the cardinality off of
      federated types.

  Returns:
    Dict of cardinalities.

  Raises:
    ValueError: If conflicting cardinalities are inferred from `value`.
    TypeError: If the arguments are of the wrong types, or if `type_spec` is
      a federated type which is not `all_equal` but the yet-to-be-embedded
      `value` is not represented as a Python `list`.
  """
  py_typecheck.check_not_none(value)
  py_typecheck.check_type(type_spec, computation_types.Type)
  if isinstance(type_spec, computation_types.FederatedType):
    if type_spec.all_equal:
      return {}
    py_typecheck.check_type(value, collections.Sized)
    return {type_spec.placement: len(value)}
  elif isinstance(type_spec, computation_types.NamedTupleType):
    anonymous_tuple_value = anonymous_tuple.from_container(
        value, recursive=False)
    cardinality_dict = {}
    for idx, (_,
              elem_type) in enumerate(anonymous_tuple.to_elements(type_spec)):
      cardinality_dict = merge_cardinalities(
          cardinality_dict,
          infer_cardinalities(anonymous_tuple_value[idx], elem_type))
    return cardinality_dict
  else:
    return {}

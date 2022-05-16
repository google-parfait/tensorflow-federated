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
"""Utilities for testing types."""

from tensorflow_federated.python.core.impl.types import computation_types


def assert_type_assignable_from(target_type, source_type):
  """Asserts that `target_type` is assignable from `source_type`."""
  message = None
  try:
    target_type.check_assignable_from(source_type)
  except computation_types.TypeNotAssignableError as e:
    message = e.message
  if message is not None:
    raise AssertionError(message)


def assert_types_equivalent(first_type, second_type):
  """Asserts that the types are equivalent."""
  message = None
  try:
    first_type.check_equivalent_to(second_type)
  except computation_types.TypesNotEquivalentError as e:
    message = e.message
  if message is not None:
    raise AssertionError(message)


def assert_types_identical(first_type, second_type):
  """Asserts that the types are identical."""
  message = None
  try:
    first_type.check_identical_to(second_type)
  except computation_types.TypesNotIdenticalError as e:
    message = e.message
  if message is not None:
    raise AssertionError(message)

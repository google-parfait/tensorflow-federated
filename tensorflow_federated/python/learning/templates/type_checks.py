# Copyright 2023, The TensorFlow Federated Authors.
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
"""Type checks for learning computations."""

from typing import Optional

import federated_language

from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_types


class ClientSequenceTypeError(Exception):
  """Raises when a type is not a structure of sequences placed at `CLIENTS`."""


# Keys of the structure that holds a tensor client dataset. See
# `model_delta_client_work.tensor_dataset_type_from_input_spec`.
TENSOR_DATASET_DATA_KEY = 'data'
TENSOR_DATASET_VALID_EXAMPLE_MASK_KEY = 'valid_example_mask'
TENSOR_DATASET_NUM_VALID_BATCHES_KEY = 'num_valid_batches'


def is_client_placed_tensor_dataset(type_spec: federated_language.Type) -> bool:
  """Returns `True` if `type_spec` is a tensor client dataset at `CLIENTS`.

  A tensor client dataset is a structure with the `TENSOR_DATASET_DATA_KEY`,
  `TENSOR_DATASET_VALID_EXAMPLE_MASK_KEY` and
  `TENSOR_DATASET_NUM_VALID_BATCHES_KEY` elements (in this order) that contains
  only tensors.

  Args:
    type_spec: A `federated_language.Type`.
  """
  if (
      not isinstance(type_spec, federated_language.FederatedType)
      or type_spec.placement is not federated_language.CLIENTS
  ):
    return False
  member = type_spec.member
  if not isinstance(member, federated_language.StructType):
    return False
  expected_names = [
      TENSOR_DATASET_DATA_KEY,
      TENSOR_DATASET_VALID_EXAMPLE_MASK_KEY,
      TENSOR_DATASET_NUM_VALID_BATCHES_KEY,
  ]
  if [name for name, _ in member.items()] != expected_names:
    return False
  return federated_language.framework.type_contains_only(
      member,
      lambda t: isinstance(
          t, (federated_language.StructType, federated_language.TensorType)
      ),
  )


def check_is_client_placed_structure_of_sequences(
    type_spec: federated_language.Type, error_message: Optional[str] = None
) -> None:
  """Checks that a type is a structure of sequences, placed at `federated_language.CLIENTS`.

  Args:
    type_spec: A `federated_language.Type`.
    error_message: An optional error message to display upon failure. If set to
      `None`, a default message is provided.

  Raises:
    ClientSequenceTypeError: If `type_spec` is not placed at
    `federated_language.CLIENTS`, or
    if its member type is not a structure of TensorFlow-compatible sequences.
  """

  def is_structure_of_sequences(member_spec: federated_language.Type) -> bool:
    if isinstance(member_spec, federated_language.SequenceType):
      return tensorflow_types.is_tensorflow_compatible_type(member_spec.element)
    elif isinstance(member_spec, federated_language.StructType):
      return all(
          is_structure_of_sequences(element_type)
          for element_type in member_spec.children()
      )
    else:
      return False

  if error_message is None:
    error_message = (
        'Expected a federated type with placement `federated_language.CLIENTS`'
        ' and with member type that is a structure of TensorFlow-compatible'
        f' sequences. Found {type_spec}.'
    )

  if (
      not isinstance(type_spec, federated_language.FederatedType)
      or type_spec.placement is not federated_language.CLIENTS
      or not is_structure_of_sequences(type_spec.member)
  ):
    raise ClientSequenceTypeError(error_message)

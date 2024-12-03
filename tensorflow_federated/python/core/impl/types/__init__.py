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
"""Libraries for interacting with the type of a computation."""
import federated_language

# pylint: disable=g-importing-member
ArrayShape = federated_language.ArrayShape
is_shape_fully_defined = federated_language.array_shape_is_fully_defined
num_elements_in_shape = federated_language.num_elements_in_array_shape
AbstractType = federated_language.AbstractType
FederatedType = federated_language.FederatedType
FunctionType = federated_language.FunctionType
PlacementType = federated_language.PlacementType
SequenceType = federated_language.SequenceType
StructType = federated_language.StructType
StructWithPythonType = federated_language.StructWithPythonType
TensorType = federated_language.TensorType
to_type = federated_language.to_type
Type = federated_language.Type
type_mismatch_error_message = (
    federated_language.framework.type_mismatch_error_message
)
TypeNotAssignableError = federated_language.framework.TypeNotAssignableError
TypeRelation = federated_language.framework.TypeRelation
TypesNotEquivalentError = federated_language.framework.TypesNotEquivalentError
TypesNotIdenticalError = federated_language.framework.TypesNotIdenticalError
UnexpectedTypeError = federated_language.framework.UnexpectedTypeError
contains = federated_language.framework.type_contains
contains_only = federated_language.framework.type_contains_only
count = federated_language.framework.type_count
is_structure_of_floats = federated_language.framework.is_structure_of_floats
is_structure_of_integers = federated_language.framework.is_structure_of_integers
is_structure_of_tensors = federated_language.framework.is_structure_of_tensors
is_tensorflow_compatible_type = (
    federated_language.framework.is_tensorflow_compatible_type
)
type_to_py_container = federated_language.framework.type_to_py_container
deserialize_type = federated_language.framework.deserialize_type
serialize_type = federated_language.framework.serialize_type
# pylint: disable=g-importing-member

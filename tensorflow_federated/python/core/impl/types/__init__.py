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

from tensorflow_federated.python.core.impl.types.array_shape import ArrayShape
from tensorflow_federated.python.core.impl.types.array_shape import is_shape_fully_defined
from tensorflow_federated.python.core.impl.types.array_shape import num_elements_in_shape
from tensorflow_federated.python.core.impl.types.computation_types import AbstractType
from tensorflow_federated.python.core.impl.types.computation_types import FederatedType
from tensorflow_federated.python.core.impl.types.computation_types import FunctionType
from tensorflow_federated.python.core.impl.types.computation_types import PlacementType
from tensorflow_federated.python.core.impl.types.computation_types import SequenceType
from tensorflow_federated.python.core.impl.types.computation_types import StructType
from tensorflow_federated.python.core.impl.types.computation_types import StructWithPythonType
from tensorflow_federated.python.core.impl.types.computation_types import tensorflow_to_type
from tensorflow_federated.python.core.impl.types.computation_types import TensorType
from tensorflow_federated.python.core.impl.types.computation_types import to_type
from tensorflow_federated.python.core.impl.types.computation_types import Type
from tensorflow_federated.python.core.impl.types.computation_types import type_mismatch_error_message
from tensorflow_federated.python.core.impl.types.computation_types import TypeNotAssignableError
from tensorflow_federated.python.core.impl.types.computation_types import TypeRelation
from tensorflow_federated.python.core.impl.types.computation_types import TypesNotEquivalentError
from tensorflow_federated.python.core.impl.types.computation_types import TypesNotIdenticalError
from tensorflow_federated.python.core.impl.types.computation_types import UnexpectedTypeError
from tensorflow_federated.python.core.impl.types.type_analysis import contains
from tensorflow_federated.python.core.impl.types.type_analysis import contains_only
from tensorflow_federated.python.core.impl.types.type_analysis import count
from tensorflow_federated.python.core.impl.types.type_analysis import is_structure_of_floats
from tensorflow_federated.python.core.impl.types.type_analysis import is_structure_of_integers
from tensorflow_federated.python.core.impl.types.type_analysis import is_structure_of_tensors
from tensorflow_federated.python.core.impl.types.type_analysis import is_tensorflow_compatible_type
from tensorflow_federated.python.core.impl.types.type_conversions import structure_from_tensor_type_tree
from tensorflow_federated.python.core.impl.types.type_conversions import type_to_py_container
from tensorflow_federated.python.core.impl.types.type_conversions import type_to_tf_tensor_specs
from tensorflow_federated.python.core.impl.types.type_serialization import deserialize_type
from tensorflow_federated.python.core.impl.types.type_serialization import serialize_type

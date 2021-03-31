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
"""Libraries for interacting with the type of a computation."""

from tensorflow_federated.python.core.api.computation_types import *
from tensorflow_federated.python.core.impl.types.type_analysis import contains
from tensorflow_federated.python.core.impl.types.type_analysis import count
from tensorflow_federated.python.core.impl.types.type_analysis import is_structure_of_tensors
from tensorflow_federated.python.core.impl.types.type_analysis import is_sum_compatible
from tensorflow_federated.python.core.impl.types.type_analysis import is_tensorflow_compatible_type
from tensorflow_federated.python.core.impl.types.type_conversions import type_from_tensors
from tensorflow_federated.python.core.impl.types.type_conversions import type_to_tf_tensor_specs
from tensorflow_federated.python.core.impl.types.type_serialization import deserialize_type
from tensorflow_federated.python.core.impl.types.type_serialization import serialize_type

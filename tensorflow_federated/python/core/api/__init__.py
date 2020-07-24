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
"""TensorFlow Federated Core API."""

from tensorflow_federated.python.core.api.computation_base import Computation
from tensorflow_federated.python.core.api.computation_types import FederatedType
from tensorflow_federated.python.core.api.computation_types import FunctionType
from tensorflow_federated.python.core.api.computation_types import SequenceType
from tensorflow_federated.python.core.api.computation_types import StructType
from tensorflow_federated.python.core.api.computation_types import StructWithPythonType
from tensorflow_federated.python.core.api.computation_types import TensorType
from tensorflow_federated.python.core.api.computation_types import to_type
from tensorflow_federated.python.core.api.computation_types import Type
from tensorflow_federated.python.core.api.computations import federated_computation
from tensorflow_federated.python.core.api.computations import tf_computation
from tensorflow_federated.python.core.api.intrinsics import federated_aggregate
from tensorflow_federated.python.core.api.intrinsics import federated_apply
from tensorflow_federated.python.core.api.intrinsics import federated_broadcast
from tensorflow_federated.python.core.api.intrinsics import federated_collect
from tensorflow_federated.python.core.api.intrinsics import federated_eval
from tensorflow_federated.python.core.api.intrinsics import federated_map
from tensorflow_federated.python.core.api.intrinsics import federated_mean
from tensorflow_federated.python.core.api.intrinsics import federated_reduce
from tensorflow_federated.python.core.api.intrinsics import federated_secure_sum
from tensorflow_federated.python.core.api.intrinsics import federated_sum
from tensorflow_federated.python.core.api.intrinsics import federated_value
from tensorflow_federated.python.core.api.intrinsics import federated_zip
from tensorflow_federated.python.core.api.intrinsics import sequence_map
from tensorflow_federated.python.core.api.intrinsics import sequence_reduce
from tensorflow_federated.python.core.api.intrinsics import sequence_sum
from tensorflow_federated.python.core.api.placements import CLIENTS
from tensorflow_federated.python.core.api.placements import SERVER
from tensorflow_federated.python.core.api.typed_object import TypedObject
from tensorflow_federated.python.core.api.value_base import Value
from tensorflow_federated.python.core.api.values import to_value

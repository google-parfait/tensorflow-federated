# Copyright 2021, The TensorFlow Federated Authors.
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
"""Python interface to C++ Value serialization implementations."""

from tensorflow_federated.cc.core.impl.executors import serialization_bindings

# Protobuf constructors.
Value = serialization_bindings.Value
Sequence = serialization_bindings.Sequence
Struct = serialization_bindings.Struct
Element = serialization_bindings.Element
Federated = serialization_bindings.Federated
Cardinality = serialization_bindings.Cardinality

# Serialization methods.
serialize_tensor_value = serialization_bindings.serialize_tensor_value
deserialize_tensor_value = serialization_bindings.deserialize_tensor_value

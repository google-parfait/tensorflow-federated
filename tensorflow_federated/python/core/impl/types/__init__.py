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
contains = federated_language.framework.type_contains
contains_only = federated_language.framework.type_contains_only
count = federated_language.framework.type_count
is_structure_of_floats = federated_language.framework.is_structure_of_floats
is_structure_of_integers = federated_language.framework.is_structure_of_integers
is_structure_of_tensors = federated_language.framework.is_structure_of_tensors
# pylint: disable=g-importing-member

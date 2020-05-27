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
"""Defines common types of placements for use in defining TFF computations."""

from tensorflow_federated.python.core.impl.types import placement_literals

# The collective of all the client devices, a TFF placement constant.
CLIENTS = placement_literals.CLIENTS

# The single top-level central coordinator, a TFF placement constant.
SERVER = placement_literals.SERVER

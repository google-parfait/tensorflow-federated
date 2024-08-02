# Copyright 2024, The TensorFlow Federated Authors.
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
"""Libraries for interacting with a TensorFlow frontend and backend."""

# pylint: disable=g-importing-member
from tensorflow_federated.python.core.environments.tensorflow_backend.tensorflow_tree_transformations import replace_intrinsics_with_bodies
from tensorflow_federated.python.core.environments.tensorflow_frontend.tensorflow_computation import tf_computation as computation
from tensorflow_federated.python.core.environments.tensorflow_frontend.tensorflow_computation import transform_args
from tensorflow_federated.python.core.environments.tensorflow_frontend.tensorflow_computation import transform_result
# pylint: enable=g-importing-member

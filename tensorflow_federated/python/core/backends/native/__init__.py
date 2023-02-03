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
"""Libraries for interacting with native backends."""

from tensorflow_federated.python.core.backends.native.compiler import desugar_and_transform_to_native
from tensorflow_federated.python.core.backends.native.compiler import transform_to_native_form
from tensorflow_federated.python.core.backends.native.execution_contexts import create_async_local_cpp_execution_context
from tensorflow_federated.python.core.backends.native.execution_contexts import create_local_python_execution_context
from tensorflow_federated.python.core.backends.native.execution_contexts import create_mergeable_comp_execution_context
from tensorflow_federated.python.core.backends.native.execution_contexts import create_remote_python_execution_context
from tensorflow_federated.python.core.backends.native.execution_contexts import create_sync_local_cpp_execution_context
from tensorflow_federated.python.core.backends.native.execution_contexts import set_async_local_cpp_execution_context
from tensorflow_federated.python.core.backends.native.execution_contexts import set_mergeable_comp_execution_context
from tensorflow_federated.python.core.backends.native.execution_contexts import set_sync_local_cpp_execution_context
from tensorflow_federated.python.core.backends.native.mergeable_comp_compiler import compile_to_mergeable_comp_form

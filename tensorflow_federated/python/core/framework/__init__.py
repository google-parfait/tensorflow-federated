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
"""Libraries for extending the TensorFlow Federated core library."""

import federated_language
# pylint: disable=g-importing-member
unique_name_generator = federated_language.framework.unique_name_generator
transform_postorder = federated_language.framework.transform_postorder
transform_preorder = federated_language.framework.transform_preorder
from tensorflow_federated.python.core.impl.compiler.transformations import to_call_dominant
from tensorflow_federated.python.core.impl.execution_contexts.mergeable_comp_execution_context import MergeableCompExecutionContext
from tensorflow_federated.python.core.impl.execution_contexts.mergeable_comp_execution_context import MergeableCompForm
from tensorflow_federated.python.core.impl.executor_stacks.executor_factory import local_cpp_executor_factory
from tensorflow_federated.python.core.impl.executor_stacks.python_executor_stacks import ResourceManagingExecutorFactory
from tensorflow_federated.python.core.impl.executors.remote_executor import RemoteExecutor
from tensorflow_federated.python.core.impl.executors.remote_executor_grpc_stub import RemoteExecutorGrpcStub
from tensorflow_federated.python.core.impl.executors.remote_executor_stub import RemoteExecutorStub
from tensorflow_federated.python.core.impl.executors.value_serialization import deserialize_value
from tensorflow_federated.python.core.impl.executors.value_serialization import serialize_value
# pylint: enable=g-importing-member

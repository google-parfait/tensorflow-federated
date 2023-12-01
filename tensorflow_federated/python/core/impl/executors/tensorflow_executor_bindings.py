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
"""Python interface to C++ Executor implementations."""

# Required to load TF Python extension.
import tensorflow as tf  # pylint: disable=unused-import

from tensorflow_federated.cc.core.impl.executors import executor_bindings

create_tensorflow_executor = executor_bindings.create_tensorflow_executor
create_dtensor_executor = executor_bindings.create_dtensor_executor

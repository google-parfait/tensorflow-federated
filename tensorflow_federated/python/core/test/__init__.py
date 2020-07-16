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
"""Test classes/functions for testing usage of TFF.

This package contains functions and utilities that can make testing usage of TFF
easier or more convenient and parallels the `tf.test` package.
"""

from tensorflow_federated.python.core.impl.reference_executor import ReferenceExecutor
from tensorflow_federated.python.core.test.static_assert import assert_contains_secure_aggregation
from tensorflow_federated.python.core.test.static_assert import assert_contains_unsecure_aggregation
from tensorflow_federated.python.core.test.static_assert import assert_not_contains_secure_aggregation
from tensorflow_federated.python.core.test.static_assert import assert_not_contains_unsecure_aggregation

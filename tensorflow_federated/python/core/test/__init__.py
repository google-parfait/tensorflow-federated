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
"""Libraries for testing TensorFlow Federated."""

from tensorflow_federated.python.core.impl.context_stack.context_stack_test_utils import with_context
from tensorflow_federated.python.core.impl.context_stack.context_stack_test_utils import with_contexts
from tensorflow_federated.python.core.impl.context_stack.runtime_error_context import create_runtime_error_context
from tensorflow_federated.python.core.impl.context_stack.set_default_context import set_no_default_context
from tensorflow_federated.python.core.impl.types.type_test_utils import assert_type_assignable_from
from tensorflow_federated.python.core.impl.types.type_test_utils import assert_types_equivalent
from tensorflow_federated.python.core.impl.types.type_test_utils import assert_types_identical
from tensorflow_federated.python.core.test.static_assert import assert_contains_secure_aggregation
from tensorflow_federated.python.core.test.static_assert import assert_contains_unsecure_aggregation
from tensorflow_federated.python.core.test.static_assert import assert_not_contains_secure_aggregation
from tensorflow_federated.python.core.test.static_assert import assert_not_contains_unsecure_aggregation

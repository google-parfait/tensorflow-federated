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
import federated_language

with_context = federated_language.framework.with_context
with_contexts = federated_language.framework.with_contexts
create_runtime_error_context = (
    federated_language.framework.create_runtime_error_context
)
set_no_default_context = federated_language.framework.set_no_default_context
assert_type_assignable_from = (
    federated_language.framework.assert_type_assignable_from
)
assert_types_equivalent = federated_language.framework.assert_types_equivalent
assert_types_identical = federated_language.framework.assert_types_identical
assert_contains_secure_aggregation = (
    federated_language.framework.assert_contains_secure_aggregation
)
assert_contains_unsecure_aggregation = (
    federated_language.framework.assert_contains_unsecure_aggregation
)
assert_not_contains_secure_aggregation = (
    federated_language.framework.assert_not_contains_secure_aggregation
)
assert_not_contains_unsecure_aggregation = (
    federated_language.framework.assert_not_contains_unsecure_aggregation
)

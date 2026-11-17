/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow_federated/cc/core/impl/aggregation/core/domain_spec.h"

#include <memory>
#include <vector>

#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated {
namespace aggregation {

DomainSpec::DomainSpec(TensorSpan domain_tensors) {
  for (const auto& tensor : domain_tensors) {
    DTYPE_CASES(
        tensor.dtype(), T,
        columns_.push_back(std::make_unique<TypedColumnDomainSpec<T>>(tensor)));
  }
}

}  // namespace aggregation
}  // namespace tensorflow_federated

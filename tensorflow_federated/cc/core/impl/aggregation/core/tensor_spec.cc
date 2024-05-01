/*
 * Copyright 2024 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

#include <utility>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"

namespace tensorflow_federated {
namespace aggregation {

StatusOr<TensorSpec> TensorSpec::FromProto(
    const TensorSpecProto& tensor_spec_proto) {
  TFF_ASSIGN_OR_RETURN(TensorShape shape,
                       TensorShape::FromProto(tensor_spec_proto.shape()));
  return TensorSpec(tensor_spec_proto.name(), tensor_spec_proto.dtype(),
                    std::move(shape));
}

TensorSpecProto TensorSpec::ToProto() const {
  TensorSpecProto tensor_spec_proto;
  tensor_spec_proto.set_name(name_);
  tensor_spec_proto.set_dtype(dtype_);
  *tensor_spec_proto.mutable_shape() = shape_.ToProto();
  return tensor_spec_proto;
}

}  // namespace aggregation
}  // namespace tensorflow_federated

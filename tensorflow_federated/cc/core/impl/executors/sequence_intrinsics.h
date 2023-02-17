/* Copyright 2021, The TensorFlow Federated Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
==============================================================================*/

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_SEQUENCE_INTRINSICS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_SEQUENCE_INTRINSICS_H_

#include <string_view>

#include "absl/status/statusor.h"

namespace tensorflow_federated {
const std::string_view kSequenceReduceUri = "sequence_reduce";
const std::string_view kSequenceMapUri = "sequence_map";

enum class SequenceIntrinsic {
  MAP,
  REDUCE,
};

absl::StatusOr<SequenceIntrinsic> SequenceIntrinsicFromUri(
    const std::string_view uri);

std::string_view SequenceIntrinsicToUri(const SequenceIntrinsic& intrinsic);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_SEQUENCE_INTRINSICS_H_

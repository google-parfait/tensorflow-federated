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

#include "tensorflow_federated/cc/core/impl/executors/sequence_intrinsics.h"

namespace tensorflow_federated {

absl::StatusOr<SequenceIntrinsic> SequenceIntrinsicFromUri(
    const absl::string_view uri) {
  if (uri == kSequenceMapUri) {
    return SequenceIntrinsic::MAP;
  } else if (uri == kSequenceReduceUri) {
    return SequenceIntrinsic::REDUCE;
  } else {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported sequence intrinsic URI: ", uri));
  }
}
}  // namespace tensorflow_federated

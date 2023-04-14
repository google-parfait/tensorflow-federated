/* Copyright 2023, The TensorFlow Federated Authors.

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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_TENSORFLOW_STATUS_COMPAT_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_TENSORFLOW_STATUS_COMPAT_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow_federated {

// On April 2023, there is not yet an official release of Tensorflow which
// includes `message().` One will need to wait for the release following 2.12.0.
// The code can be updated to just be the else branch after such release exists.
absl::string_view ToMessage(tsl::Status tf_status);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_TENSORFLOW_STATUS_COMPAT_H_

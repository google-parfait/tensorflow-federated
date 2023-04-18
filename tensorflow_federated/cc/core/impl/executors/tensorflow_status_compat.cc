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

#include "tensorflow_federated/cc/core/impl/executors/tensorflow_status_compat.h"

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow_federated {

absl::string_view ToMessage(tsl::Status tf_status) {
#if TF_GRAPH_DEF_VERSION < 1467
  return tf_status.error_message();
#else
  return tf_status.message();
#endif
}

}  // namespace tensorflow_federated

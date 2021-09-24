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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_COMPUTATIONS_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_COMPUTATIONS_H_

#include <string>

#include "tensorflow_federated/proto/v0/computation.pb.h"

namespace tensorflow_federated {

inline v0::Computation IdentityComp() {
  v0::Computation comp;
  v0::Lambda* lambda = comp.mutable_lambda();
  *lambda->mutable_parameter_name() = "x";
  *lambda->mutable_result()->mutable_reference()->mutable_name() = "x";
  return comp;
}

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_COMPUTATIONS_H_

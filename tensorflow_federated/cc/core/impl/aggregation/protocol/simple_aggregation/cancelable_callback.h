/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_CANCELABLE_CALLBACK_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_CANCELABLE_CALLBACK_H_

#include <functional>
#include <memory>

#include "absl/time/time.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/clock.h"

namespace tensorflow_federated::aggregation {

// Abstract interface for an object that can be canceled.
class Cancelable {
 public:
  virtual ~Cancelable() = default;
  virtual void Cancel() = 0;
};

using CancelationToken = std::shared_ptr<Cancelable>;

// Schedules a delayed callback that can be canceled by calling Cancel method on
// the CancelationToken.
CancelationToken ScheduleCallback(Clock* clock, absl::Duration delay,
                                  std::function<void()> callback);

}  // namespace tensorflow_federated::aggregation

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_PROTOCOL_SIMPLE_AGGREGATION_CANCELABLE_CALLBACK_H_

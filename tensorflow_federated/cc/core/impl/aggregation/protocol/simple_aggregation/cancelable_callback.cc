// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow_federated/cc/core/impl/aggregation/protocol/simple_aggregation/cancelable_callback.h"

#include <functional>
#include <memory>
#include <utility>

#include "absl/time/time.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/clock.h"

namespace tensorflow_federated::aggregation {

namespace {

class CancelableWaiter : public Cancelable, public Clock::Waiter {
 public:
  explicit CancelableWaiter(std::function<void()> callback)
      : callback_(std::move(callback)) {}
  ~CancelableWaiter() override = default;

 private:
  void Cancel() override {
    absl::MutexLock lock(&mu_);
    callback_ = nullptr;
  }

  void WakeUp() override {
    absl::MutexLock lock(&mu_);
    if (callback_ != nullptr) {
      callback_();
      callback_ = nullptr;
    }
  }

  absl::Mutex mu_;
  std::function<void()> callback_ ABSL_GUARDED_BY(mu_);
};

}  // namespace

CancelationToken ScheduleCallback(Clock* clock, absl::Duration delay,
                                  std::function<void()> callback) {
  auto waiter = std::make_shared<CancelableWaiter>(std::move(callback));
  clock->WakeupWithDeadline(clock->Now() + delay, waiter);
  return waiter;
}

}  // namespace tensorflow_federated::aggregation

/*
 * Copyright 2020 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/base/simulated_clock.h"

namespace tensorflow_federated {

absl::Time SimulatedClock::Now() {
  absl::MutexLock lock(mutex());
  return NowLocked();
}

absl::Time SimulatedClock::NowLocked() {
  mutex()->AssertHeld();
  return now_;
}

void SimulatedClock::Sleep(absl::Duration d) {
  absl::Time current;
  {
    absl::MutexLock lock(mutex());
    current = now_;
  }
  absl::Time deadline = current + d;
  while (true) {
    {
      absl::MutexLock lock(mutex());
      current = now_;
    }
    if (current >= deadline) {
      return;
    }
  }
}

void SimulatedClock::SetTime(absl::Time t) {
  {
    absl::MutexLock lock(mutex());
    now_ = t;
  }
  DispatchWakeups();
}

void SimulatedClock::AdvanceTime(absl::Duration d) {
  {
    absl::MutexLock lock(mutex());
    now_ += d;
  }
  DispatchWakeups();
}

}  // namespace tensorflow_federated

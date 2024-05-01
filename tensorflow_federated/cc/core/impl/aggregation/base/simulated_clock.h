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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_BASE_SIMULATED_CLOCK_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_BASE_SIMULATED_CLOCK_H_

#include "tensorflow_federated/cc/core/impl/aggregation/base/clock.h"

namespace tensorflow_federated {

/*
 * A simulated clock is a concrete Clock implementation that does not "tick"
 * on its own.  Time is advanced by explicit calls to the AdvanceTime() or
 * SetTime() functions.
 */
class SimulatedClock : public Clock {
 public:
  // Construct SimulatedClock with a specific initial time.
  explicit SimulatedClock(absl::Time t) : now_(t) {}

  // Construct SimulatedClock with default initial time (1970-01-01 00:00:00)
  SimulatedClock() : SimulatedClock(absl::UnixEpoch()) {}

  // Returns the simulated time.
  absl::Time Now() override;

  // Sleeps until the specified duration has elapsed according to this clock.
  void Sleep(absl::Duration d) override;

  // Sets the simulated time. Wakes up any waiters whose deadlines have now
  // expired.
  void SetTime(absl::Time t);

  // Advances the simulated time. Wakes up any waiters whose deadlines have now
  // expired.
  void AdvanceTime(absl::Duration d);

 private:
  // Returns the simulated time (called internally from the base class).
  absl::Time NowLocked() override;

  // No specific scheduling is needed for SimulatedClock.
  void ScheduleWakeup(absl::Time wakeup_time) override {}

  absl::Time now_ ABSL_GUARDED_BY(mutex());
};

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_BASE_SIMULATED_CLOCK_H_

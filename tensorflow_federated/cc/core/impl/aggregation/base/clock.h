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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_BASE_CLOCK_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_BASE_CLOCK_H_

#include <map>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace tensorflow_federated {

/*
 * Clock is an abstract class representing a Clock, which is an object that can
 * tell you the current time and schedule a wakeable event in a future.
 */
class Clock {
 public:
  // Returns a pointer to the global real time clock. The caller does not
  // own the returned pointer and should not delete it. The returned clock
  // is thread-safe.
  static Clock* RealClock();

  virtual ~Clock() = default;

  // Returns current time.
  virtual absl::Time Now() = 0;

  // Sleeps for the specified duration.
  virtual void Sleep(absl::Duration d) = 0;

  // An abstract interface for a waiter class that is passed to
  // WakeupWithDeadline and is responsible for handling a timer wake-up.
  // Waiter interface doesn't support a cancellation mechanism which means
  //
  // Note: it is up to Waiter implementation how to handle a cancellation. Clock
  // itself doesn't manage cancellation and will call WakeUp() on all all alarms
  // once their deadline time is past due.
  class Waiter {
   public:
    virtual ~Waiter() = default;
    // A callback method that is called when the corresponding deadline is
    // reached. This method may be called on an arbitrary thread.
    virtual void WakeUp() = 0;
  };

  // Schedule the waiter to be waked up at the specified deadline.
  void WakeupWithDeadline(absl::Time deadline,
                          const std::shared_ptr<Waiter>& waiter);

 protected:
  // Accessors shared for derived classes.
  absl::Mutex* mutex() { return &mu_; }

  // Internal version of now which is called under mutex.
  virtual absl::Time NowLocked()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex()) = 0;

  // Overloaded by derived class to implement the actual scheduling.
  virtual void ScheduleWakeup(absl::Time wakeup_time)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex()) = 0;

  // Called to dispatch wakeup to all due waiters.
  void DispatchWakeups();

 private:
  using WaiterList = std::vector<std::shared_ptr<Waiter>>;
  using WaiterMap = std::map<absl::Time, WaiterList>;

  bool CheckReentrancy();
  WaiterList GetExpiredWaiters();
  bool FinishDispatchAndScheduleNextWakeup();

  // Mutex that protects the internal state.
  absl::Mutex mu_;
  // Pending (unexpired) waiters ordered by deadline - soonest to latest.
  // Waiters with exactly the same deadline are stored in the same bucket and
  // the order at which they were added is preserved.
  WaiterMap pending_waiters_ ABSL_GUARDED_BY(mutex());
  // This value =0 when no DispatchWakeups() is running;
  //            =1 when DispatchWakeups() is running
  //            >1 when at least one additional DispatchWakeups() call happened
  //               while DispatchWakeups() was running, for example from
  //               a timer elapsing and triggering a wake-up.
  int dispatch_level_ ABSL_GUARDED_BY(mutex()) = 0;
};

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_BASE_CLOCK_H_

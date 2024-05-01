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

#include "tensorflow_federated/cc/core/impl/aggregation/base/clock.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"

namespace tensorflow_federated {

// Implements global realtime clock that uses timers to schedule wake-up of
// waiters.
class RealTimeClock : public Clock {
 public:
  RealTimeClock();
  ~RealTimeClock() override {
    TFF_LOG(FATAL) << "RealTimeClock should never be destroyed";
  }

  // Returns the current time.
  absl::Time Now() override { return absl::Now(); }
  absl::Time NowLocked() override { return absl::Now(); }

  // Sleeps for the specified duration.
  void Sleep(absl::Duration d) override { absl::SleepFor(d); }

  // Schedules wakeup at the specified wakeup_time.
  void ScheduleWakeup(absl::Time wakeup_time) override;

 private:
  // The worker loop that performs the sleep and dispatches wake-up calls.
  void WorkerLoop();

  // The currently scheduled wake-up time. There is at most one wake-up
  // time per process.
  absl::Time next_wakeup_ ABSL_GUARDED_BY(&wakeup_mu_) = absl::InfiniteFuture();
  // Mutex that protects next_wakeup and used with the wake-up CondVar.
  absl::Mutex wakeup_mu_;
  // CondVar used to sleep until the next wake-up deadline.
  absl::CondVar wakeup_condvar_;
  // Worker thread that runs the worker loop function. Since this class
  // is singleton, there is only one thread per process, and that thread is
  // never terminated.
  std::unique_ptr<std::thread> worker_thread_;
};

Clock* Clock::RealClock() {
  static Clock* realtime_clock = new RealTimeClock;
  return realtime_clock;
}

void Clock::WakeupWithDeadline(absl::Time deadline,
                               const std::shared_ptr<Clock::Waiter>& waiter) {
  // Insert the new waiter into the map ordered by its deadline.
  // Waiters with matching deadlines are inserted into the same bucket and
  // their order within the bucket is preserved.
  {
    absl::MutexLock lock(mutex());
    WaiterMap::iterator it;
    if ((it = pending_waiters_.find(deadline)) != pending_waiters_.end()) {
      it->second.push_back(waiter);
    } else {
      pending_waiters_.insert(std::make_pair(deadline, WaiterList{waiter}));
    }
  }

  // Inserting a new waiter may trigger an immediate wake-up if the deadline
  // is due. Otherwise a new wake-up is scheduled at the end on the dispatch.
  DispatchWakeups();
}

// DispatchWakeup performs the following actions in the loop:
// - Check for reentrancy to avoid more than one concurrent dispatch loop
// - Take out all waiters that are due
// - Make WakeUp calls on all of those waiters. This step is done outside
//   of lock because it may potentially take longer time and new waiters may
//   potentially be inserted during that step.
// - If there are any waiters that are still due at that point (because the
//   the previous step took too long and new waiters have expired or because
//   there were any new waiters inserted during the previous steps), loop
//   back and repeat the previous steps.
// - Otherwise finish the dispatch by scheduling a new wakeup for the bucket
//   that expires the soonest.
void Clock::DispatchWakeups() {
  do {
    if (CheckReentrancy()) {
      // Avoid reentrancy. An ongoing DispatchWakeups() call will take care
      // of dispatching any new due wakeups if necessary.
      // If there is a race condition, only one of dispatch calls will go
      // through and all other will just increment the dispatch_level and
      // return.
      return;
    }

    // Collect waiters that are due.
    WaiterList wakeup_calls = GetExpiredWaiters();

    // Dispatch WakeUp calls to those waiters.
    for (const auto& waiter : wakeup_calls) {
      waiter->WakeUp();
    }
    // One more dispatch loop may be needed if there were any reentrant calls
    // or if WakeUp() calls took so long that more waiters have become due.
  } while (!FinishDispatchAndScheduleNextWakeup());
}

// Called at the beginning of dispatch loop.
// Increments dispatch_level_ and returns true if there is already
// another dispatch call in progress.
bool Clock::CheckReentrancy() {
  absl::MutexLock lock(mutex());
  return ++dispatch_level_ > 1;
}

// Iterate through waiter buckets ordered by deadline time and take out all
// waiters that are due.
Clock::WaiterList Clock::GetExpiredWaiters() {
  absl::MutexLock lock(mutex());
  absl::Time now = NowLocked();
  std::vector<std::shared_ptr<Waiter>> wakeup_calls;
  WaiterMap::iterator iter;

  while ((iter = pending_waiters_.begin()) != pending_waiters_.end() &&
         iter->first <= now) {
    std::move(iter->second.begin(), iter->second.end(),
              std::back_inserter(wakeup_calls));
    pending_waiters_.erase(iter);
  }
  return wakeup_calls;
}

// Called at the end of dispatch loop to check post-dispatch conditions,
// reset re-entrancy level, and schedule a next dispatch if needed.
// Returns true if the dispatch loop has ended.
// Returns false if more the dispatch loop needs to be repeated.
bool Clock::FinishDispatchAndScheduleNextWakeup() {
  absl::MutexLock lock(mutex());
  int current_dispatch_level = dispatch_level_;
  dispatch_level_ = 0;

  if (!pending_waiters_.empty()) {
    if (current_dispatch_level > 1) {
      // There was another dispatch call while this one was in progress.
      // One more dispatch loop is needed.
      return false;
    }

    absl::Time next_wakeup = pending_waiters_.begin()->first;
    if (next_wakeup <= NowLocked()) {
      // One more dispatch loop is needed because a new waiter has become due
      // while the wake-ups were dispatched.
      return false;
    }

    // Schedule DispatchWakeups() to be called at a future next_wakeup time.
    ScheduleWakeup(next_wakeup);
  }

  return true;
}

RealTimeClock::RealTimeClock() {
  worker_thread_ =
      std::make_unique<std::thread>([this] { this->WorkerLoop(); });
}

void RealTimeClock::WorkerLoop() {
  for (;;) {
    bool dispatch = false;

    {
      absl::MutexLock lock(&wakeup_mu_);
      wakeup_condvar_.WaitWithDeadline(&wakeup_mu_, next_wakeup_);
      if (Now() >= next_wakeup_) {
        dispatch = true;
        next_wakeup_ = absl::InfiniteFuture();
      }
    }

    if (dispatch) {
      DispatchWakeups();
    }
  }
}

// RealTimeClock implementation of ScheduleWakeup.
void RealTimeClock::ScheduleWakeup(absl::Time wakeup_time) {
  absl::MutexLock lock(&wakeup_mu_);

  // Optimization: round wakeup_time up to whole milliseconds.
  wakeup_time = absl::FromUDate(ceil(absl::ToUDate(wakeup_time)));

  // ScheduleWakeup may be called repeatedly with the same time if a new timer
  // is created in the future after already existing timer. In that case
  // this function continues relying on already scheduled wake-up time.
  // A new ScheduleWakeup call will be made from within DispatchWakeups() once
  // the current timer expires.
  if (wakeup_time == next_wakeup_) {
    return;
  }

  next_wakeup_ = wakeup_time;
  wakeup_condvar_.Signal();
}

}  // namespace tensorflow_federated

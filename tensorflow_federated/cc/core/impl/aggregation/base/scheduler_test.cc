/*
 * Copyright 2018 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/base/scheduler.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "googletest/include/gtest/gtest.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tensorflow_federated {
namespace base {
namespace {

// NOTE: many of tests below use reference captures in lambdas for locals.
// This is sound because the test methods do not return before the thread
// pool has become idle (pool->WaitUntilIdle()).

// Tests whether scheduled tasks are successfully executed.
TEST(ThreadPool, TasksAreExecuted) {
  auto pool = CreateThreadPoolScheduler(2);

  bool b1 = false;
  bool b2 = false;
  pool->Schedule([&b1]() { b1 = true; });
  pool->Schedule([&b2]() { b2 = true; });

  pool->WaitUntilIdle();

  EXPECT_TRUE(b1);
  EXPECT_TRUE(b2);
}

// Tests whether the pool uses actually multiple threads to execute tasks.
// The test goal is achieved by blocking in one task until another task
// unblocks, which can only work if multiple threads are used.
TEST(ThreadPool, ThreadsAreUtilized) {
  auto pool = CreateThreadPoolScheduler(2);

  absl::BlockingCounter counter(1);
  bool b1 = false;
  bool b2 = false;

  pool->Schedule([&b1, &counter] {
    counter.Wait();
    b1 = true;
  });
  pool->Schedule([&b2, &counter] {
    counter.DecrementCount();
    b2 = true;
  });

  pool->WaitUntilIdle();

  EXPECT_TRUE(b1);
  EXPECT_TRUE(b2);
}

TEST(ThreadPool, StressTest) {
  // A simple stress test where we spawn many threads and let them after
  // a random wait time increment a counter.
  static constexpr int kThreads = 32;
  static constexpr int kIterations = 16;
  auto pool = CreateThreadPoolScheduler(kThreads);
  std::atomic<int64_t> atomic_counter{0};

  for (auto i = 0; i < kThreads; ++i) {
    auto task = [&atomic_counter] {
      for (auto j = 0; j < kIterations; ++j) {
        absl::SleepFor(absl::Microseconds(std::rand() % 500));
        atomic_counter.fetch_add(1);
      }
    };
    pool->Schedule(task);
  }

  pool->WaitUntilIdle();
  ASSERT_EQ(atomic_counter, kThreads * kIterations);
}

TEST(Worker, TasksAreExecutedSequentially) {
  auto pool = CreateThreadPoolScheduler(3);
  auto worker = pool->CreateWorker();
  absl::Mutex mutex{};
  std::vector<int> recorded{};
  for (int i = 0; i < 128; i++) {
    worker->Schedule([&mutex, &recorded, i] {
      // Expect that no one is holding the mutex (tests for non-overlap).
      if (mutex.TryLock()) {
        // Add i to the recorded values (tests for execution in order).
        recorded.push_back(i);
        // Idle wait to be sure we don't execute faster than we schedule
        absl::SleepFor(absl::Milliseconds(50));
        mutex.Unlock();
      } else {
        FAIL() << "mutex was unexpectedly hold";
      }
    });
  }
  pool->WaitUntilIdle();

  // Verify recorded values.
  for (int i = 0; i < 128; i++) {
    ASSERT_EQ(recorded[i], i);
  }
}

}  // namespace

}  // namespace base
}  // namespace tensorflow_federated

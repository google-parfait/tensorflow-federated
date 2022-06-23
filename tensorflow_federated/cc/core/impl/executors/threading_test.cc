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

#include "tensorflow_federated/cc/core/impl/executors/threading.h"

#include <atomic>
#include <cstdint>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/synchronization/barrier.h"
#include "absl/time/time.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated {

namespace {

using ::absl::StatusCode;

class ThreadingTest : public ::testing::Test {};

TEST_F(ThreadingTest, ParallelTasksEmptyIsOk) {
  ParallelTasks tasks;
  EXPECT_THAT(tasks.WaitAll(), IsOk());
}

TEST_F(ThreadingTest, ParallelTasksSingleTaskIsOk) {
  ParallelTasks tasks;
  tasks.add_task([]() { return absl::OkStatus(); });
  EXPECT_THAT(tasks.WaitAll(), IsOk());
}

TEST_F(ThreadingTest, ParallelTasksSingleTaskError) {
  ParallelTasks tasks;
  tasks.add_task([]() { return absl::UnimplementedError(""); });
  EXPECT_THAT(tasks.WaitAll(), StatusIs(StatusCode::kUnimplemented));
}

TEST_F(ThreadingTest, ParallelTasksOneOkOneErrorIsError) {
  ParallelTasks tasks;
  tasks.add_task([]() { return absl::OkStatus(); });
  tasks.add_task([]() { return absl::UnimplementedError(""); });
  EXPECT_THAT(tasks.WaitAll(), StatusIs(StatusCode::kUnimplemented));
}

TEST_F(ThreadingTest, ParallelTasksWaitsForAll) {
  constexpr int32_t NUM_TASKS = 5000;
  std::atomic<int32_t> counter(0);
  ParallelTasks tasks;
  absl::Barrier barrier(NUM_TASKS + 1);
  for (int32_t i = 0; i < NUM_TASKS; i++) {
    tasks.add_task([&counter, &barrier]() {
      barrier.Block();  // Block until all threads have hit here.
      counter.fetch_add(1);
      return absl::OkStatus();
    });
  }
  // Since all tasks will block until the main thread also blocks on the
  // barrier, we can assert they are all still 0 here. If new threads were not
  // spawned the test would timeout above, being blocked.
  EXPECT_THAT(counter.load(), 0);
  barrier.Block();  // Release the threads to increment the counter.
  EXPECT_THAT(tasks.WaitAll(), IsOk());
  EXPECT_EQ(counter.load(), NUM_TASKS);
}

TEST_F(ThreadingTest, ParallelTasksDebugNoNewThreads) {
  constexpr int32_t NUM_TASKS = 10;
  std::atomic<int32_t> counter(0);
  ParallelTasks tasks(/*debug_mode=*/true);
  for (int32_t i = 0; i < NUM_TASKS; i++) {
    EXPECT_THAT(tasks.WaitAll(), IsOk());
    tasks.add_task([&counter]() {
      counter.fetch_add(1);
      return absl::OkStatus();
    });
    // Assert that the counter was incremented (task was run immediately)
    // every loop iteration. This is non-deterministic when debug_mode is false.
    EXPECT_THAT(counter.load(), i);
    EXPECT_THAT(tasks.WaitAll(), IsOk());
  }
  EXPECT_THAT(tasks.WaitAll(), IsOk());
  EXPECT_EQ(counter.load(), NUM_TASKS);
}

}  // namespace

}  // namespace tensorflow_federated

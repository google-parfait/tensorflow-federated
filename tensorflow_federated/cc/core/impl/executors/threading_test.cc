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
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/threadpool.h"
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
  const int32_t NUM_TASKS = 5000;
  std::atomic<int32_t> counter(0);
  ParallelTasks tasks;
  for (int32_t i = 0; i < NUM_TASKS; i++) {
    tasks.add_task([&counter]() {
      counter.fetch_add(1);
      return absl::OkStatus();
    });
  }
  EXPECT_THAT(tasks.WaitAll(), IsOk());
  EXPECT_EQ(counter.load(), NUM_TASKS);
}

TEST_F(ThreadingTest, ParallelTasksWithThreadPool) {
  const int32_t NUM_THREADS = 3;
  tensorflow::thread::ThreadPool thread_pool(tensorflow::Env::Default(),
                                             "test_thread_pool", NUM_THREADS);
  std::atomic<int32_t> counter(0);
  absl::Notification event;
  ParallelTasks tasks(&thread_pool);
  for (int32_t i = 0; i < NUM_THREADS * 2; i++) {
    tasks.add_task([&counter, &event]() {
      counter.fetch_add(1);
      event.WaitForNotification();
      return absl::OkStatus();
    });
  }
  // Sleep a few seconds to ensure NUM_THREADS have run.
  absl::SleepFor(absl::Seconds(10));
  // Assert that only NUM_THREADS have run, as they are waiting for notification
  // to complete before the threadpool can schedule the remaining threads.
  EXPECT_EQ(counter.load(), NUM_THREADS);
  // Notify the threads to complete, freeing the threadpool to run the remaining
  // tasks.
  event.Notify();
  EXPECT_THAT(tasks.WaitAll(), IsOk());
  EXPECT_EQ(counter.load(), NUM_THREADS * 2);
}

}  // namespace

}  // namespace tensorflow_federated

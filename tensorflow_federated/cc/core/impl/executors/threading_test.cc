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
#include <memory>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated {

namespace {

using ::absl::StatusCode;

class ThreadPoolDeathTest : public ::testing::Test {};

TEST_F(ThreadPoolDeathTest, NumThreadsZero) {
  ASSERT_DEATH(ThreadPool(/*num_threads=*/0, /*name=*/"test"),
               "num_threads must be positive");
}

TEST_F(ThreadPoolDeathTest, NumThreadsNegative) {
  ASSERT_DEATH(ThreadPool(/*num_threads=*/-1, /*name=*/"test"),
               "num_threads must be positive");
}

class ThreadPoolTest : public ::testing::Test {};

TEST_F(ThreadPoolTest, SingleThreadFIFO) {
  constexpr int32_t NUM_WORK = 10;
  ThreadPool pool(/*num_threads=*/1, /*name=*/"test");
  std::vector<int32_t> results;
  results.reserve(NUM_WORK);
  absl::BlockingCounter blocking_counter(NUM_WORK);
  for (int i = 0; i < NUM_WORK; ++i) {
    TFF_ASSERT_OK(pool.Schedule([&results, &blocking_counter, i]() {
      results.push_back(i);
      blocking_counter.DecrementCount();
    }));
  }
  blocking_counter.Wait();
  EXPECT_THAT(results, testing::ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
}

TEST_F(ThreadPoolTest, DependentWorkScheduledInSeparateThread) {
  ThreadPool pool(/*num_threads=*/2, /*name=*/"test");
  absl::Notification event;
  std::vector<int32_t> results;
  absl::BlockingCounter counter(2);
  TFF_ASSERT_OK(pool.Schedule([&event, &counter, &results]() {
    event.WaitForNotification();
    results.push_back(1);
    counter.DecrementCount();
  }));
  TFF_ASSERT_OK(pool.Schedule([&event, &counter, &results]() {
    results.push_back(2);
    event.Notify();
    counter.DecrementCount();
  }));
  counter.Wait();
  EXPECT_THAT(results, testing::ElementsAre(2, 1));
}

TEST_F(ThreadPoolTest, MultipleThreads) {
  constexpr int32_t NUM_WORK = 100;
  ThreadPool pool(/*num_threads=*/10, /*name=*/"test");
  absl::Mutex results_mutex;
  std::vector<int32_t> results ABSL_GUARDED_BY(results_mutex), expected_results;
  results.reserve(NUM_WORK);
  expected_results.reserve(NUM_WORK);
  absl::BlockingCounter blocking_counter(NUM_WORK);
  for (int i = 0; i < NUM_WORK; ++i) {
    expected_results.push_back(i);
    TFF_ASSERT_OK(
        pool.Schedule([&results, &results_mutex, &blocking_counter, i]() {
          {
            absl::MutexLock lock(&results_mutex);
            results.push_back(i);
          }
          blocking_counter.DecrementCount();
        }));
  }
  blocking_counter.Wait();
  ASSERT_THAT(results, testing::UnorderedElementsAreArray(expected_results));
}

TEST_F(ThreadPoolTest, ShuttingDownPoolErrorsOnSchedule) {
  ThreadPool pool(/*num_threads=*/1, /*name=*/"test");
  pool.Close();
  ASSERT_THAT(pool.Schedule([]() {}),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       testing::HasSubstr("closed")));
}

class ParallelTasksTest : public ::testing::Test {};

TEST_F(ParallelTasksTest, EmptyIsOk) {
  ParallelTasks tasks;
  EXPECT_THAT(tasks.WaitAll(), IsOk());
}

TEST_F(ParallelTasksTest, SingleTaskIsOk) {
  ParallelTasks tasks;
  TFF_ASSERT_OK(tasks.add_task([]() { return absl::OkStatus(); }));
  EXPECT_THAT(tasks.WaitAll(), IsOk());
}

TEST_F(ParallelTasksTest, ParallelTasksSingleTaskError) {
  ParallelTasks tasks;
  TFF_ASSERT_OK(tasks.add_task([]() { return absl::UnimplementedError(""); }));
  EXPECT_THAT(tasks.WaitAll(), StatusIs(StatusCode::kUnimplemented));
}

TEST_F(ParallelTasksTest, OneOkOneErrorIsError) {
  ParallelTasks tasks;
  TFF_ASSERT_OK(tasks.add_task([]() { return absl::OkStatus(); }));
  TFF_ASSERT_OK(tasks.add_task([]() { return absl::UnimplementedError(""); }));
  EXPECT_THAT(tasks.WaitAll(), StatusIs(StatusCode::kUnimplemented));
}

TEST_F(ParallelTasksTest, ParallelTasksWaitsForAll) {
  const int32_t NUM_TASKS = 5000;
  std::atomic<int32_t> counter(0);
  ParallelTasks tasks;
  for (int32_t i = 0; i < NUM_TASKS; i++) {
    TFF_ASSERT_OK(tasks.add_task([&counter]() {
      counter.fetch_add(1);
      return absl::OkStatus();
    }));
  }
  EXPECT_THAT(tasks.WaitAll(), IsOk());
  EXPECT_EQ(counter.load(), NUM_TASKS);
}

TEST_F(ParallelTasksTest, WithThreadPool) {
  const int32_t NUM_THREADS = 3;
  ThreadPool thread_pool(NUM_THREADS, /*name=*/"test");
  std::atomic<int32_t> counter(0);
  absl::Notification event;
  ParallelTasks tasks(&thread_pool);
  for (int32_t i = 0; i < NUM_THREADS * 2; i++) {
    TFF_ASSERT_OK(tasks.add_task([&counter, &event]() {
      counter.fetch_add(1);
      event.WaitForNotification();
      return absl::OkStatus();
    }));
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

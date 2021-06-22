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

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"

namespace tensorflow_federated {

namespace {

using ::absl::StatusCode;
using ::testing::status::IsOk;
using ::testing::status::StatusIs;

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
  const uint32 NUM_TASKS = 5000;
  std::atomic<uint32> counter = 0;
  ParallelTasks tasks;
  for (uint32 i = 0; i < NUM_TASKS; i++) {
    tasks.add_task([&counter]() {
      counter.fetch_add(1);
      return absl::OkStatus();
    });
  }
  EXPECT_THAT(tasks.WaitAll(), IsOk());
  EXPECT_EQ(counter.load(), NUM_TASKS);
}

}  // namespace

}  // namespace tensorflow_federated

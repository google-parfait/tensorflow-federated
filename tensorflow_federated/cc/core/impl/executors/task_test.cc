/* Copyright 2026, The TensorFlow Federated Authors.

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

#include "tensorflow_federated/cc/core/impl/executors/task.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {

namespace {

using ::tensorflow_federated::IsOk;
using ::tensorflow_federated::IsOkAndHolds;
using ::tensorflow_federated::StatusIs;

TEST(TaskTest, EvaluatesLazilyAndReturnsValue) {
  bool executed = false;
  auto MakeTask = [&]() -> Task<int> {
    executed = true;
    co_return 42;
  };

  Task<int> task = MakeTask();
  EXPECT_FALSE(executed);
  EXPECT_FALSE(task.is_ready());

  int result = SyncWait(std::move(task));
  EXPECT_TRUE(executed);
  EXPECT_EQ(result, 42);
}

TEST(TaskTest, VoidTaskExecutes) {
  bool executed = false;
  auto MakeTask = [&]() -> Task<void> {
    executed = true;
    co_return;
  };

  Task<void> task = MakeTask();
  EXPECT_FALSE(executed);
  SyncWait(std::move(task));
  EXPECT_TRUE(executed);
}

TEST(TaskTest, PropagatesMoveOnlyType) {
  auto MakeChild = []() -> Task<std::unique_ptr<int>> {
    co_return std::make_unique<int>(123);
  };

  auto MakeParent = [&]() -> Task<std::unique_ptr<int>> {
    std::unique_ptr<int> ptr = co_await MakeChild();
    *ptr += 1;
    co_return ptr;
  };

  std::unique_ptr<int> result = SyncWait(MakeParent());
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(*result, 124);
}

TEST(TaskTest, DeepChainDoesNotOverflowStack) {
  constexpr int kChainDepth = 10000;
  auto RecursiveStep = [](auto& self, int depth) -> Task<int> {
    if (depth == 0) {
      co_return 0;
    }
    int child_val = co_await self(self, depth - 1);
    co_return child_val + 1;
  };

  int result = SyncWait(RecursiveStep(RecursiveStep, kChainDepth));
  EXPECT_EQ(result, kChainDepth);
}

TEST(SharedTaskTest, MemoizesResultAndProvidesConstRef) {
  int execution_count = 0;
  auto Compute = [&]() -> SharedTask<int> {
    execution_count++;
    co_return 100;
  };

  SharedTask<int> shared = Compute();
  EXPECT_EQ(execution_count, 1);

  const int& val1 = SyncWait(shared);
  const int& val2 = SyncWait(shared);
  EXPECT_EQ(val1, 100);
  EXPECT_EQ(val2, 100);
  EXPECT_EQ(&val1, &val2);
  EXPECT_EQ(execution_count, 1);
}

TEST(SharedTaskTest, VoidSharedTaskExecutes) {
  bool executed = false;
  auto Compute = [&]() -> SharedTask<void> {
    executed = true;
    co_return;
  };

  SharedTask<void> shared = Compute();
  EXPECT_TRUE(executed);
  SyncWait(shared);

  SharedTask<void> ready = SharedTask<void>::MakeReady();
  EXPECT_TRUE(ready.is_ready());
  SyncWait(ready);
}

TEST(SharedTaskTest, MultipleConsumersInspectMoveOnlyValue) {
  auto Compute = []() -> SharedTask<std::unique_ptr<int>> {
    co_return std::make_unique<int>(999);
  };

  SharedTask<std::unique_ptr<int>> shared = Compute();

  auto Consumer1 = [](SharedTask<std::unique_ptr<int>> s) -> Task<int> {
    const std::unique_ptr<int>& ptr = co_await s;
    co_return *ptr + 1;
  };

  auto Consumer2 = [](SharedTask<std::unique_ptr<int>> s) -> Task<int> {
    const std::unique_ptr<int>& ptr = co_await s;
    co_return *ptr + 2;
  };

  int r1 = SyncWait(Consumer1(shared));
  int r2 = SyncWait(Consumer2(shared));
  EXPECT_EQ(r1, 1000);
  EXPECT_EQ(r2, 1001);
}

TEST(SharedTaskTest, ConcurrentConsumersWaitAndResume) {
  constexpr int kNumConsumers = 10;
  absl::Notification trigger;

  auto Producer = [&]() -> Task<int> {
    trigger.WaitForNotification();
    co_return 777;
  };

  ThreadPool pool(/*num_threads=*/4, /*name=*/"producer_pool");
  SharedTask<int> shared = ScheduleTask(&pool, Producer());

  std::vector<Task<int>> consumer_tasks;
  consumer_tasks.reserve(kNumConsumers);
  for (int i = 0; i < kNumConsumers; ++i) {
    consumer_tasks.emplace_back([](SharedTask<int> s) -> Task<int> {
      const int& val = co_await s;
      co_return val;
    }(shared));
  }

  std::vector<SharedTask<int>> consumer_shared;
  consumer_shared.reserve(kNumConsumers);
  for (Task<int>& ct : consumer_tasks) {
    consumer_shared.emplace_back(ScheduleTask(&pool, std::move(ct)));
  }

  trigger.Notify();

  for (const SharedTask<int>& cs : consumer_shared) {
    EXPECT_EQ(SyncWait(cs), 777);
  }
}

TEST(SharedTaskTest, DeepChainDoesNotOverflowStack) {
  constexpr int kChainDepth = 10000;
  std::vector<SharedTask<int>> tasks;
  tasks.reserve(kChainDepth);

  tasks.push_back(SharedTask<int>::FromValue(0));
  for (int i = 1; i < kChainDepth; ++i) {
    tasks.push_back(MakeShared([](SharedTask<int> prev) -> Task<int> {
      const int& val = co_await prev;
      co_return val + 1;
    }(tasks.back())));
  }

  EXPECT_EQ(SyncWait(tasks.back()), kChainDepth - 1);
}

TEST(ScheduleTaskTest, SingleThreadWorkerDoesNotDeadlockOnChainedAwaits) {
  ThreadPool single_thread_pool(/*num_threads=*/1, /*name=*/"single_worker");

  absl::Notification b_can_start;
  auto TaskB = [&]() -> Task<int> {
    b_can_start.WaitForNotification();
    co_return 42;
  };

  SharedTask<int> b = ScheduleTask(&single_thread_pool, TaskB());

  auto TaskA = [&](SharedTask<int> dep) -> Task<int> {
    const int& b_val = co_await dep;
    co_return b_val * 2;
  };

  SharedTask<int> a = ScheduleTask(&single_thread_pool, TaskA(b));

  b_can_start.Notify();

  EXPECT_EQ(SyncWait(a), 84);
}

TEST(CoroutineStatusMacrosTest, ReturnIfErrorAndAssignOrReturn) {
  auto ReturnError = []() -> Task<absl::Status> {
    co_return absl::InvalidArgumentError("bad argument");
  };

  auto ReturnOk = []() -> Task<absl::Status> { co_return absl::OkStatus(); };

  auto CallWithStatus = [&](bool fail) -> Task<absl::Status> {
    if (fail) {
      TFF_CO_RETURN_IF_ERROR(co_await ReturnError());
    } else {
      TFF_CO_RETURN_IF_ERROR(co_await ReturnOk());
    }
    co_return absl::OkStatus();
  };

  EXPECT_THAT(SyncWait(CallWithStatus(true)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(SyncWait(CallWithStatus(false)), IsOk());

  auto MakeStatusOr =
      [](bool fail) -> Task<absl::StatusOr<std::unique_ptr<int>>> {
    if (fail) {
      co_return absl::NotFoundError("not found");
    }
    co_return std::make_unique<int>(555);
  };

  auto UnpackStatusOr = [&](bool fail) -> Task<absl::StatusOr<int>> {
    TFF_CO_ASSIGN_OR_RETURN(std::unique_ptr<int> val,
                            co_await MakeStatusOr(fail));
    co_return *val + 1;
  };

  EXPECT_THAT(SyncWait(UnpackStatusOr(true)),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(SyncWait(UnpackStatusOr(false)), IsOkAndHolds(556));

  auto BindStatusOr = [&](SharedTask<absl::StatusOr<std::unique_ptr<int>>> s)
      -> Task<absl::StatusOr<int>> {
    TFF_CO_BIND_OR_RETURN(const std::unique_ptr<int>& val, co_await s);
    co_return *val + 10;
  };

  SharedTask<absl::StatusOr<std::unique_ptr<int>>> shared_val =
      MakeShared(MakeStatusOr(false));
  EXPECT_THAT(SyncWait(BindStatusOr(shared_val)), IsOkAndHolds(565));
}

TEST(MakeReadySharedTaskTest, ImmediatelyReadyWithoutSuspension) {
  SharedTask<int> ready_int = MakeReadySharedTask(42);
  EXPECT_TRUE(ready_int.is_ready());
  EXPECT_EQ(SyncWait(ready_int), 42);

  SharedTask<void> ready_void = MakeReadySharedTask();
  EXPECT_TRUE(ready_void.is_ready());
  SyncWait(ready_void);

  struct MoveOnlyStruct {
    std::unique_ptr<int> val;
    explicit MoveOnlyStruct(int v) : val(std::make_unique<int>(v)) {}
    MoveOnlyStruct(MoveOnlyStruct&&) = default;
    MoveOnlyStruct& operator=(MoveOnlyStruct&&) = default;
    MoveOnlyStruct(const MoveOnlyStruct&) = delete;
    MoveOnlyStruct& operator=(const MoveOnlyStruct&) = delete;
  };

  SharedTask<MoveOnlyStruct> ready_move_only =
      MakeReadySharedTask(MoveOnlyStruct(123));
  EXPECT_TRUE(ready_move_only.is_ready());
  const MoveOnlyStruct& ref = SyncWait(ready_move_only);
  EXPECT_EQ(*ref.val, 123);
}

}  // namespace

}  // namespace tensorflow_federated

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

#include "tensorflow_federated/cc/core/impl/executors/disposal_queue.h"

#include <cstddef>
#include <memory>
#include <string>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "include/grpcpp/server_context.h"
#include "include/grpcpp/support/status.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_grpc.h"
#include "tensorflow_federated/proto/v0/executor.grpc.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {
namespace {

using ::testing::_;

constexpr char kExecutorId[] = "test_executor";

v0::ValueRef MakeValueRef(const std::string& id) {
  v0::ValueRef ref;
  ref.set_id(id);
  return ref;
}

class DisposalQueueTest : public ::testing::Test {
 protected:
  DisposalQueueTest() {
    mock_service_ = mock_server_.service();
    stub_ = mock_server_.NewStub();
    executor_pb_.set_id(kExecutorId);
  }

  MockGrpcExecutorServer mock_server_;
  MockGrpcExecutorService* mock_service_;
  std::shared_ptr<v0::ExecutorGroup::StubInterface> stub_;
  v0::ExecutorId executor_pb_;
};

TEST_F(DisposalQueueTest, UsesExpectedDefaultParameters) {
  EXPECT_EQ(DisposalQueue::kDefaultMaxBatchSize, 256);
  EXPECT_EQ(DisposalQueue::kDefaultFlushDelay, absl::Milliseconds(10));
}

TEST_F(DisposalQueueTest, DispatchesImmediatelyWhenBufferingDisabled) {
  constexpr size_t kBatchSize = 0;
  std::shared_ptr<DisposalQueue> queue = std::make_shared<DisposalQueue>(
      executor_pb_, stub_, kBatchSize, absl::Minutes(10));

  absl::Notification notified;
  EXPECT_CALL(*mock_service_, Dispose(_, _, _))
      .WillOnce([&](grpc::ServerContext*, const v0::DisposeRequest* request,
                    v0::DisposeResponse*) {
        EXPECT_EQ(request->executor().id(), kExecutorId);
        EXPECT_EQ(request->value_ref().size(), 1);
        EXPECT_EQ(request->value_ref(0).id(), "val_unbuffered");
        notified.Notify();
        return grpc::Status::OK;
      });

  queue->Dispose(MakeValueRef("val_unbuffered"));

  ASSERT_TRUE(notified.WaitForNotificationWithTimeout(absl::Seconds(5)));
}

TEST_F(DisposalQueueTest, DispatchesBatchWhenThresholdExceeded) {
  constexpr size_t kBatchSize = 4;
  std::shared_ptr<DisposalQueue> queue = std::make_shared<DisposalQueue>(
      executor_pb_, stub_, kBatchSize, absl::Minutes(10));

  absl::Notification notified;
  EXPECT_CALL(*mock_service_, Dispose(_, _, _))
      .WillOnce([&](grpc::ServerContext*, const v0::DisposeRequest* request,
                    v0::DisposeResponse*) {
        EXPECT_EQ(request->executor().id(), kExecutorId);
        EXPECT_EQ(request->value_ref().size(), kBatchSize);
        for (int i = 0; i < kBatchSize; ++i) {
          EXPECT_EQ(request->value_ref(i).id(), absl::StrCat("val_", i));
        }
        notified.Notify();
        return grpc::Status::OK;
      });

  for (int i = 0; i < kBatchSize; ++i) {
    queue->Dispose(MakeValueRef(absl::StrCat("val_", i)));
  }

  ASSERT_TRUE(notified.WaitForNotificationWithTimeout(absl::Seconds(5)));
}

TEST_F(DisposalQueueTest, DispatchesBatchWhenTimerExpires) {
  constexpr size_t kBatchSize = 100;
  constexpr absl::Duration kDelay = absl::Milliseconds(20);
  std::shared_ptr<DisposalQueue> queue =
      std::make_shared<DisposalQueue>(executor_pb_, stub_, kBatchSize, kDelay);

  absl::Notification notified;
  EXPECT_CALL(*mock_service_, Dispose(_, _, _))
      .WillOnce([&](grpc::ServerContext*, const v0::DisposeRequest* request,
                    v0::DisposeResponse*) {
        EXPECT_EQ(request->executor().id(), kExecutorId);
        EXPECT_EQ(request->value_ref().size(), 2);
        EXPECT_EQ(request->value_ref(0).id(), "val_0");
        EXPECT_EQ(request->value_ref(1).id(), "val_1");
        notified.Notify();
        return grpc::Status::OK;
      });

  queue->Dispose(MakeValueRef("val_0"));
  queue->Dispose(MakeValueRef("val_1"));

  ASSERT_TRUE(notified.WaitForNotificationWithTimeout(absl::Seconds(5)));
}

TEST_F(DisposalQueueTest, FlushesExplicitly) {
  constexpr size_t kBatchSize = 100;
  std::shared_ptr<DisposalQueue> queue = std::make_shared<DisposalQueue>(
      executor_pb_, stub_, kBatchSize, absl::Minutes(10));

  absl::Notification notified;
  EXPECT_CALL(*mock_service_, Dispose(_, _, _))
      .WillOnce([&](grpc::ServerContext*, const v0::DisposeRequest* request,
                    v0::DisposeResponse*) {
        EXPECT_EQ(request->value_ref().size(), 1);
        EXPECT_EQ(request->value_ref(0).id(), "val_explicit");
        notified.Notify();
        return grpc::Status::OK;
      });

  queue->Dispose(MakeValueRef("val_explicit"));
  queue->Flush();

  ASSERT_TRUE(notified.WaitForNotificationWithTimeout(absl::Seconds(5)));
}

TEST_F(DisposalQueueTest, ClosesWithoutDispatchingPending) {
  constexpr size_t kBatchSize = 100;
  std::shared_ptr<DisposalQueue> queue = std::make_shared<DisposalQueue>(
      executor_pb_, stub_, kBatchSize, absl::Minutes(10));

  EXPECT_CALL(*mock_service_, Dispose(_, _, _)).Times(0);

  queue->Dispose(MakeValueRef("val_drop"));
  queue->Close();
  queue->Flush();
  queue->Dispose(MakeValueRef("val_after_close"));
}

TEST_F(DisposalQueueTest, HandlesGrpcErrorGracefully) {
  constexpr size_t kBatchSize = 2;
  std::shared_ptr<DisposalQueue> queue = std::make_shared<DisposalQueue>(
      executor_pb_, stub_, kBatchSize, absl::Minutes(10));

  absl::Notification notified;
  EXPECT_CALL(*mock_service_, Dispose(_, _, _))
      .WillOnce([&](grpc::ServerContext*, const v0::DisposeRequest*,
                    v0::DisposeResponse*) {
        notified.Notify();
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Server down");
      });

  queue->Dispose(MakeValueRef("val_0"));
  queue->Dispose(MakeValueRef("val_1"));

  ASSERT_TRUE(notified.WaitForNotificationWithTimeout(absl::Seconds(5)));
}

}  // namespace
}  // namespace tensorflow_federated

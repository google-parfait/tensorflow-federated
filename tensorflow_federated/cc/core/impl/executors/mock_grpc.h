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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_MOCK_GRPC_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_MOCK_GRPC_H_

#include <memory>

#include "googlemock/include/gmock/gmock.h"
#include "absl/strings/str_cat.h"
#include "include/grpcpp/grpcpp.h"
#include "include/grpcpp/security/credentials.h"
#include "include/grpcpp/security/server_credentials.h"
#include "include/grpcpp/support/status.h"
#include "third_party/py/federated_language_executor/executor.grpc.pb.h"
#include "third_party/py/federated_language_executor/executor.pb.h"

namespace tensorflow_federated {

class MockGrpcExecutorService
    : public federated_language_executor::ExecutorGroup::Service {
 public:
  MOCK_METHOD(grpc::Status, GetExecutor,
              (grpc::ServerContext*,
               const federated_language_executor::GetExecutorRequest*,
               federated_language_executor::GetExecutorResponse*));
  MOCK_METHOD(grpc::Status, CreateValue,
              (grpc::ServerContext*,
               const federated_language_executor::CreateValueRequest*,
               federated_language_executor::CreateValueResponse*));
  MOCK_METHOD(grpc::Status, CreateCall,
              (grpc::ServerContext*,
               const federated_language_executor::CreateCallRequest*,
               federated_language_executor::CreateCallResponse*));
  MOCK_METHOD(grpc::Status, CreateStruct,
              (grpc::ServerContext*,
               const federated_language_executor::CreateStructRequest*,
               federated_language_executor::CreateStructResponse*));
  MOCK_METHOD(grpc::Status, CreateSelection,
              (grpc::ServerContext*,
               const federated_language_executor::CreateSelectionRequest*,
               federated_language_executor::CreateSelectionResponse*));
  MOCK_METHOD(grpc::Status, Compute,
              (grpc::ServerContext*,
               const federated_language_executor::ComputeRequest*,
               federated_language_executor::ComputeResponse*));
  MOCK_METHOD(grpc::Status, Dispose,
              (grpc::ServerContext*,
               const federated_language_executor::DisposeRequest*,
               federated_language_executor::DisposeResponse*));
  MOCK_METHOD(grpc::Status, DisposeExecutor,
              (grpc::ServerContext*,
               const federated_language_executor::DisposeExecutorRequest*,
               federated_language_executor::DisposeExecutorResponse*));
};

// A minimal, self-contained, OSS-compatible mock GRPC Executor service.
//
// This is an alternative Google-internal
// `//net/grpc/testing/mocker/mock_grpc.h`, which is not usable in OSS.
class MockGrpcExecutorServer {
 public:
  explicit MockGrpcExecutorServer()
      : server_(grpc::ServerBuilder()
                    .AddListeningPort(
                        "localhost:0",
                        grpc::experimental::LocalServerCredentials(LOCAL_TCP),
                        &port_)
                    .RegisterService(&service_)
                    .BuildAndStart()) {}

  ~MockGrpcExecutorServer() {
    server_->Shutdown();
    server_->Wait();
  }

  MockGrpcExecutorService* service() { return &service_; }

  std::unique_ptr<federated_language_executor::ExecutorGroup::Stub> NewStub() {
    return federated_language_executor::ExecutorGroup::NewStub(
        grpc::CreateChannel(absl::StrCat("localhost:", port_),
                            grpc::experimental::LocalCredentials(LOCAL_TCP)));
  }

 private:
  MockGrpcExecutorService service_;
  int port_ = 0;
  const std::unique_ptr<grpc::Server> server_;
};

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_MOCK_GRPC_H_

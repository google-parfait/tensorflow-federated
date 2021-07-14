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

#include "net/grpc/public/include/grpcpp/grpcpp.h"
#include "googlemock/include/gmock/gmock.h"
#include "tensorflow_federated/proto/v0/executor.grpc.pb.h"
#include "tensorflow_federated/proto/v0/executor.proto.h"

namespace tensorflow_federated {

class MockGrpcExecutorService : public v0::Executor::Service {
 public:
  MOCK_METHOD(grpc::Status, CreateValue,
              (grpc::ServerContext*, const v0::CreateValueRequest*,
               v0::CreateValueResponse*));
  MOCK_METHOD(grpc::Status, CreateCall,
              (grpc::ServerContext*, const v0::CreateCallRequest*,
               v0::CreateCallResponse*));
  MOCK_METHOD(grpc::Status, CreateStruct,
              (grpc::ServerContext*, const v0::CreateStructRequest*,
               v0::CreateStructResponse*));
  MOCK_METHOD(grpc::Status, CreateSelection,
              (grpc::ServerContext*, const v0::CreateSelectionRequest*,
               v0::CreateSelectionResponse*));
  MOCK_METHOD(grpc::Status, Compute,
              (grpc::ServerContext*, const v0::ComputeRequest*,
               v0::ComputeResponse*));
  MOCK_METHOD(grpc::Status, SetCardinalities,
              (grpc::ServerContext*, const v0::SetCardinalitiesRequest*,
               v0::SetCardinalitiesResponse*));
  MOCK_METHOD(grpc::Status, ClearExecutor,
              (grpc::ServerContext*, const v0::ClearExecutorRequest*,
               v0::ClearExecutorResponse*));
  MOCK_METHOD(grpc::Status, Dispose,
              (grpc::ServerContext*, const v0::DisposeRequest*,
               v0::DisposeResponse*));
};

// A minimal, self-contained, OSS-compatible mock GRPC Executor service.
//
// This is an alternative Google-internal
// `//net/grpc/testing/mocker/mock_grpc.h`, which is not usable in OSS.
class MockGrpcExecutorServer {
 public:
  explicit MockGrpcExecutorServer()
      : server_(ABSL_DIE_IF_NULL(
            grpc::ServerBuilder()
                .AddListeningPort(
                    "localhost:0",
                    grpc::experimental::LocalServerCredentials(LOCAL_TCP),
                    &port_)
                .RegisterService(&service_)
                .BuildAndStart())) {}

  ~MockGrpcExecutorServer() {
    server_->Shutdown();
    server_->Wait();
  }

  MockGrpcExecutorService* service() { return &service_; }

  std::unique_ptr<v0::Executor::Stub> NewStub() {
    return v0::Executor::NewStub(
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

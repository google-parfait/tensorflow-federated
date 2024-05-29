/* Copyright 2022, The TensorFlow Federated Authors.

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

#include "tensorflow_federated/cc/core/impl/executors/dtensor_api.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/log/check.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/dtensor/cc/mesh_type.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/proto/layout.pb.h"
#include "tsl/platform/statusor.h"

namespace tensorflow_federated {
namespace dtensor {
namespace {
using ::testing::HasSubstr;

tensorflow::StatusOr<tensorflow::dtensor::Layout> ShardedOnFirstDimLayout(
    int rank, const tensorflow::dtensor::Mesh& mesh) {
  std::vector<std::string> sharding_specs;
  sharding_specs.push_back("x");
  for (int i = 1; i < rank; ++i) {
    sharding_specs.push_back(tensorflow::dtensor::Layout::kUnshardedDim);
  }
  return tensorflow::dtensor::Layout::GetLayout(sharding_specs, mesh);
}

tensorflow::dtensor::MeshProto CreateMeshForTest() {
  tensorflow::dtensor::MeshProto mesh;

  tensorflow::dtensor::MeshDimensionProto* dimension =
      mesh.add_mesh_dimensions();
  dimension->set_name("x");
  dimension->set_size(2);
  mesh.add_local_devices("/job:localhost/replica:0/task:0/device:CPU:0");
  mesh.add_local_devices("/job:localhost/replica:0/task:0/device:CPU:1");
  mesh.add_global_devices("/job:localhost/replica:0/task:0/device:CPU:0");
  mesh.add_global_devices("/job:localhost/replica:0/task:0/device:CPU:0");
  mesh.add_local_device_ids(0);
  mesh.add_local_device_ids(1);
  mesh.add_global_device_ids(0);
  mesh.add_global_device_ids(1);
  return mesh;
}

tensorflow::Tensor CreateIntTensor(tensorflow::TensorShape shape,
                                   const std::vector<int32_t>& elements) {
  CHECK(shape.num_elements() == elements.size());
  tensorflow::Tensor tensor(tensorflow::DT_INT32, shape);
  auto flat = tensor.flat<int32_t>();
  for (size_t i = 0; i < elements.size(); i++) {
    flat(i) = elements[i];
  }
  return tensor;
}

class DTensorAPITest : public ::testing::Test {
 public:
  DTensorAPITest() {
    TF_Status* status = TF_NewStatus();
    std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)>
        opts(TFE_NewContextOptions(), TFE_DeleteContextOptions);
    std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
        TFE_NewContext(opts.get(), status), TFE_DeleteContext);

    TFE_SetLogicalCpuDevices(context.get(), 2,
                             "/job:localhost/replica:0/task:0", status);
    device_name_ = "/job:localhost/replica:0/task:0/device:CUSTOM:1";
    TF_DeleteStatus(status);
  }

  std::string device_name_;
};

TEST_F(DTensorAPITest, CheckTensorDTensorWithShardedLayout) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);
  TFE_SetLogicalCpuDevices(context.get(), 2, "/job:localhost/replica:0/task:0",
                           status);

  TF_ASSERT_OK_AND_ASSIGN(auto mesh, tensorflow::dtensor::Mesh::ParseFromProto(
                                         CreateMeshForTest()));
  TFE_DTENSOR_RegisterDTensorDevice(context.get(), tensorflow::wrap(&mesh),
                                    device_name_.c_str(), status);

  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TF_ASSERT_OK_AND_ASSIGN(auto layout, ShardedOnFirstDimLayout(1, mesh));

  tensorflow::Tensor tensor =
      CreateIntTensor(tensorflow::TensorShape({4}), {1, 2, 3, 4});
  TFE_TensorHandle* tensor_handle = TFE_NewTensorHandle(tensor, status);

  TFE_TensorHandle* dtensor_handle = TFE_DTENSOR_TensorToDTensor(
      context.get(), tensor_handle, tensorflow::wrap(&layout),
      device_name_.c_str(), status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  tensorflow::ImmediateExecutionTensorHandle* dtensor =
      tensorflow::unwrap(dtensor_handle);
  std::string summary;
  ASSERT_TRUE(dtensor->SummarizeValue(summary).ok());
  EXPECT_THAT(summary, AllOf(HasSubstr("{\"CPU:0\": [1 2], \"CPU:1\": [3 4]}"),
                             HasSubstr("x")));
  EXPECT_THAT(dtensor->DebugString(),
              AllOf(HasSubstr("dtype=DT_INT32"),
                    HasSubstr("{\"CPU:0\": [1 2], \"CPU:1\": [3 4]}"),
                    HasSubstr("sharding_specs:x")));

  TFE_TensorHandle* converted_tensor_handle = TFE_DTENSOR_DTensorToTensor(
      context.get(), dtensor_handle, device_name_.c_str(), status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> value_tensor(
      TFE_TensorHandleResolve(converted_tensor_handle, status),
      TF_DeleteTensor);

  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  int32_t expected[4];
  memcpy(&expected[0], TF_TensorData(value_tensor.get()),
         TF_TensorByteSize(value_tensor.get()));
  EXPECT_EQ(1, expected[0]);
  EXPECT_EQ(2, expected[1]);
  EXPECT_EQ(3, expected[2]);
  EXPECT_EQ(4, expected[3]);

  TF_DeleteStatus(status);
  TFE_DeleteTensorHandle(tensor_handle);
  TFE_DeleteTensorHandle(dtensor_handle);
  TFE_DeleteTensorHandle(converted_tensor_handle);
}

TEST_F(DTensorAPITest, CheckCopyToMesh) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);
  TFE_SetLogicalCpuDevices(context.get(), 2, "/job:localhost/replica:0/task:0",
                           status);

  TF_ASSERT_OK_AND_ASSIGN(auto mesh, tensorflow::dtensor::Mesh::ParseFromProto(
                                         CreateMeshForTest()));
  TFE_DTENSOR_RegisterDTensorDevice(context.get(), tensorflow::wrap(&mesh),
                                    device_name_.c_str(), status);

  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  auto layout = tensorflow::dtensor::Layout::ReplicatedOnMesh(mesh, 1);

  tensorflow::Tensor tensor =
      CreateIntTensor(tensorflow::TensorShape({4}), {1, 2, 3, 4});
  TFE_TensorHandle* tensor_handle = TFE_NewTensorHandle(tensor, status);

  TFE_TensorHandle* dtensor_handle = TFE_DTENSOR_CopyToMesh(
      context.get(), tensor_handle, tensorflow::wrap(&layout),
      device_name_.c_str(), status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  tensorflow::ImmediateExecutionTensorHandle* dtensor =
      tensorflow::unwrap(dtensor_handle);
  std::string summary;
  ASSERT_TRUE(dtensor->SummarizeValue(summary).ok());
  EXPECT_THAT(summary, AllOf(HasSubstr("[1 2 3 4]"), HasSubstr("unsharded")));
  EXPECT_THAT(dtensor->DebugString(),
              AllOf(HasSubstr("dtype=DT_INT32"), HasSubstr("[1 2 3 4]"),
                    HasSubstr("sharding_specs:unsharded")));

  // Reading tensor value from replicated DTensor is allowed.
  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> value_tensor(
      TFE_TensorHandleResolve(dtensor_handle, status), TF_DeleteTensor);

  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  int32_t expected[4];
  memcpy(&expected[0], TF_TensorData(value_tensor.get()),
         TF_TensorByteSize(value_tensor.get()));
  EXPECT_EQ(1, expected[0]);
  EXPECT_EQ(2, expected[1]);
  EXPECT_EQ(3, expected[2]);
  EXPECT_EQ(4, expected[3]);

  TF_DeleteStatus(status);
  TFE_DeleteTensorHandle(tensor_handle);
  TFE_DeleteTensorHandle(dtensor_handle);
}

TEST_F(DTensorAPITest, CheckRelayout) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);
  TFE_SetLogicalCpuDevices(context.get(), 2, "/job:localhost/replica:0/task:0",
                           status);
  TF_ASSERT_OK_AND_ASSIGN(auto mesh, tensorflow::dtensor::Mesh::ParseFromProto(
                                         CreateMeshForTest()));
  TFE_DTENSOR_RegisterDTensorDevice(context.get(), tensorflow::wrap(&mesh),
                                    device_name_.c_str(), status);

  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  tensorflow::Tensor tensor =
      CreateIntTensor(tensorflow::TensorShape({4}), {1, 2, 3, 4});
  TFE_TensorHandle* tensor_handle = TFE_NewTensorHandle(tensor, status);

  auto replicated_layout =
      tensorflow::dtensor::Layout::ReplicatedOnMesh(mesh, 1);
  TFE_TensorHandle* dtensor_handle = TFE_DTENSOR_CopyToMesh(
      context.get(), tensor_handle, tensorflow::wrap(&replicated_layout),
      device_name_.c_str(), status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  tensorflow::ImmediateExecutionTensorHandle* dtensor =
      tensorflow::unwrap(dtensor_handle);
  std::string summary;
  ASSERT_TRUE(dtensor->SummarizeValue(summary).ok());
  EXPECT_THAT(summary, AllOf(HasSubstr("[1 2 3 4]"), HasSubstr("unsharded")));
  EXPECT_THAT(dtensor->DebugString(),
              AllOf(HasSubstr("dtype=DT_INT32"), HasSubstr("[1 2 3 4]"),
                    HasSubstr("sharding_specs:unsharded")));

  TF_ASSERT_OK_AND_ASSIGN(auto layout, ShardedOnFirstDimLayout(1, mesh));
  TFE_TensorHandle* relayout_dtensor_handle = TFE_DTENSOR_Relayout(
      context.get(), dtensor_handle, tensorflow::wrap(&layout),
      device_name_.c_str(), status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  tensorflow::ImmediateExecutionTensorHandle* dtensor1 =
      tensorflow::unwrap(relayout_dtensor_handle);
  std::string summary1;
  ASSERT_TRUE(dtensor1->SummarizeValue(summary1).ok());
  EXPECT_THAT(summary1, AllOf(HasSubstr("{\"CPU:0\": [1 2], \"CPU:1\": [3 4]}"),
                              HasSubstr("x")));
  EXPECT_THAT(dtensor1->DebugString(),
              AllOf(HasSubstr("dtype=DT_INT32"),
                    HasSubstr("{\"CPU:0\": [1 2], \"CPU:1\": [3 4]}"),
                    HasSubstr("sharding_specs:x")));

  TF_DeleteStatus(status);
  TFE_DeleteTensorHandle(tensor_handle);
  TFE_DeleteTensorHandle(dtensor_handle);
  TFE_DeleteTensorHandle(relayout_dtensor_handle);
}
}  // namespace
}  // namespace dtensor
}  // namespace tensorflow_federated

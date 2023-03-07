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
#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DTENSOR_EXECUTOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DTENSOR_EXECUTOR_H_

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/dtensor/cc/mesh_type.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"

namespace tensorflow_federated {
// A converter interface introduced for testing so that the API can be mocked.
class DTensorConverter {
 public:
  virtual ~DTensorConverter() = default;
  // Wrapper for Tensor to DTensor conversion
  virtual TFE_TensorHandle* TensorToDTensor(TFE_Context* context,
                                            TFE_TensorHandle* tensor_handle,
                                            const tensorflow::TF_Layout* layout,
                                            const char* device_name,
                                            TF_Status* status) = 0;
  // Wrapper for DTensor to Tensor conversion
  virtual TFE_TensorHandle* DTensorToTensor(TFE_Context* context,
                                            TFE_TensorHandle* tensor_handle,
                                            const char* device_name,
                                            TF_Status* status) = 0;
};

// Returns an executor that will use provided device for tensorflow computation.
// The device_name can be a registered DTensor device.
// max_concurrent_computation_calls can be used to control maximum number
// of active threads executing tensorflow functions.
std::shared_ptr<Executor> CreateDTensorExecutor(
    TFE_Context* context, std::optional<std::string> dtensor_device_name,
    std::optional<tensorflow::dtensor::Mesh> mesh = std::nullopt,
    std::unique_ptr<DTensorConverter> dtensor_converter =
        nullptr,  // Uses
                  // default when null.
                  //  Used only when mesh is
                  // specified.
    int32_t max_concurrent_computation_calls = -1);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DTENSOR_EXECUTOR_H_

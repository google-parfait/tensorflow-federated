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
#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DTENSOR_API_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DTENSOR_API_H_

#include <string>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/dtensor/cc/mesh_type.h"

extern "C" {

// Registers a DTensor device with provided mesh.
// Returns a DeviceInfo object which can be used to add mesh
void* TFE_DTENSOR_RegisterDTensorDevice(TFE_Context* context,
                                        tensorflow::TF_Mesh* mesh,
                                        const char* dtensor_device_name,
                                        TF_Status* status);

// Returns true, if given tensor_handle points to a DTensor on provided device
// name.
bool TFE_DTENSOR_IsTensorHandleOnDevice(TFE_Context* context,
                                        TFE_TensorHandle* tensor_handle,
                                        const char* device_name,
                                        TF_Status* status);

// Copies a Tensor to DTensor by sharding or replicating the input tensor
// according to specified layout.
TFE_TensorHandle* TFE_DTENSOR_TensorToDTensor(
    TFE_Context* context, TFE_TensorHandle* tensor_handle,
    const tensorflow::TF_Layout* layout, const char* device_name,
    TF_Status* status);

// Copies input DTensor to Tensor, by removing the sharding and
// returns the global tensor value handle.
TFE_TensorHandle* TFE_DTENSOR_DTensorToTensor(TFE_Context* context,
                                              TFE_TensorHandle* dtensor_handle,
                                              const char* device_name,
                                              TF_Status* status);

// Copies a Tensor onto mesh with replicated layout and returns DTensor.
// CopyToMesh only supports replicated layout.
// Input handle to CopyToMesh is expected to be a regular tensor.
TFE_TensorHandle* TFE_DTENSOR_CopyToMesh(TFE_Context* context,
                                         TFE_TensorHandle* tensor_handle,
                                         const tensorflow::TF_Layout* layout,
                                         const char* device_name,
                                         TF_Status* status);

// Changes the layout of input DTensor to provided layout and returns resulting
// DTensor handle.
// Note that input handle is expected to be DTensor handle, passing a regular
// tensor to Relayout will result in a invalid argument status.
// TODO(b/256948367): Relayout does not support complex dtypes and some dtypes
// on GPU. Add documentation on supported types and fix the support for dtypes.
TFE_TensorHandle* TFE_DTENSOR_Relayout(TFE_Context* context,
                                       TFE_TensorHandle* handle,
                                       const tensorflow::TF_Layout* layout,
                                       const char* device_name,
                                       TF_Status* status);

} /* end extern "C" */

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_DTENSOR_API_H_

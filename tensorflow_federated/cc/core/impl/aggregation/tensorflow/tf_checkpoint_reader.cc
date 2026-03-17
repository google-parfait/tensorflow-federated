/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Vendored copy of tensorflow/c/checkpoint_reader.cc.

#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/tf_checkpoint_reader.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace checkpoint {

class TensorSliceReader;

CheckpointReader::CheckpointReader(const string& filename, TF_Status* status)
    : reader_(nullptr),
      v2_reader_(nullptr),
      var_to_shape_map_(nullptr),
      var_to_data_type_map_(nullptr) {
  // Depending on whether this is a V2 ckpt, initializes "reader_" or
  // "v2_reader_".
  std::vector<std::string> v2_path;
  if (Env::Default()->GetMatchingPaths(MetaFilename(filename), &v2_path).ok() &&
      !v2_path.empty()) {
    v2_reader_ = std::make_unique<BundleReader>(
        Env::Default(), filename /* prefix to a V2 ckpt */);
    if (!v2_reader_->status().ok()) {
      tsl::Set_TF_Status_from_Status(status, v2_reader_->status());
      return;
    }
    auto result = BuildV2VarMaps();
    var_to_shape_map_.swap(result.first);
    var_to_data_type_map_.swap(result.second);
  } else {
    reader_ = std::make_unique<TensorSliceReader>(filename);
    if (!reader_->status().ok()) {
      tsl::Set_TF_Status_from_Status(status, reader_->status());
      return;
    }
    var_to_shape_map_ = std::make_unique<TensorSliceReader::VarToShapeMap>(
        reader_->GetVariableToShapeMap());
    var_to_data_type_map_ =
        std::make_unique<TensorSliceReader::VarToDataTypeMap>(
            reader_->GetVariableToDataTypeMap());
  }
}

bool CheckpointReader::HasTensor(const std::string& name) const {
  if (reader_ != nullptr) {
    return reader_->HasTensor(name, nullptr, nullptr);
  }
  return v2_reader_->Contains(name);
}

const TensorSliceReader::VarToShapeMap&
CheckpointReader::GetVariableToShapeMap() const {
  CHECK(var_to_shape_map_);
  return *var_to_shape_map_;
}

const TensorSliceReader::VarToDataTypeMap&
CheckpointReader::GetVariableToDataTypeMap() const {
  CHECK(var_to_data_type_map_);
  return *var_to_data_type_map_;
}

std::string CheckpointReader::DebugString() const {
  if (reader_ != nullptr) return reader_->DebugString();
  return v2_reader_->DebugString();
}

void CheckpointReader::GetTensor(
    const std::string& name, std::unique_ptr<tensorflow::Tensor>* out_tensor,
    TF_Status* out_status) const {
  absl::Status status;
  if (reader_ != nullptr) {
    status = reader_->GetTensor(name, out_tensor);
  } else {
    tensorflow::DataType dtype;
    tensorflow::TensorShape shape;
    status = v2_reader_->LookupDtypeAndShape(name, &dtype, &shape);
    if (status.ok()) {
      *out_tensor = std::make_unique<tensorflow::Tensor>(dtype, shape);
      status = v2_reader_->Lookup(name, out_tensor->get());
      if (!status.ok()) out_tensor->reset();
    }
  }
  if (!status.ok()) {
    tsl::Set_TF_Status_from_Status(out_status, status);
  }
}

std::pair<std::unique_ptr<TensorSliceReader::VarToShapeMap>,
          std::unique_ptr<TensorSliceReader::VarToDataTypeMap>>
CheckpointReader::BuildV2VarMaps() {
  CHECK(v2_reader_ != nullptr);
  CHECK_OK(v2_reader_->status());

  // First pass: filters out the entries of the slices.
  absl::flat_hash_set<std::string> filtered_keys;
  BundleEntryProto entry;
  v2_reader_->Seek(kHeaderEntryKey);
  for (v2_reader_->Next(); v2_reader_->Valid(); v2_reader_->Next()) {
    CHECK(entry.ParseFromString(v2_reader_->value()))
        << entry.InitializationErrorString();
    for (int i = 0; i < entry.slices_size(); ++i) {
      const auto& slice_proto = entry.slices(i);
      CHECK(filtered_keys
                .insert(EncodeTensorNameSlice(
                    string(v2_reader_->key()) /* full var's name */,
                    TensorSlice(slice_proto)))
                .second);
    }
  }

  // Second pass: adds the entries, ignoring the filtered keys.
  auto var_to_shape_map = std::make_unique<TensorSliceReader::VarToShapeMap>();
  auto var_to_data_type_map =
      std::make_unique<TensorSliceReader::VarToDataTypeMap>();
  v2_reader_->Seek(kHeaderEntryKey);
  for (v2_reader_->Next(); v2_reader_->Valid(); v2_reader_->Next()) {
    if (filtered_keys.count(v2_reader_->key()) > 0) continue;
    CHECK(entry.ParseFromString(v2_reader_->value()))
        << entry.InitializationErrorString();
    std::string key(v2_reader_->key());
    (*var_to_shape_map)[key] = TensorShape(entry.shape());
    (*var_to_data_type_map)[key] = DataType(entry.dtype());
  }
  // The returned pointers are owned by the caller.
  return std::make_pair(std::move(var_to_shape_map),
                        std::move(var_to_data_type_map));
}

}  // namespace checkpoint
}  // namespace tensorflow

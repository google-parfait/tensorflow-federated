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

// Vendored copy of tensorflow/c/checkpoint_reader.cc, moved into the
// tensorflow_federated::aggregation::checkpoint namespace to avoid duplicate
// symbol errors when both this and the original TF version are linked.

#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/tf_checkpoint_reader.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow_federated {
namespace aggregation {
namespace checkpoint {

namespace tf = ::tensorflow;
namespace tf_ckpt = ::tensorflow::checkpoint;

CheckpointReader::CheckpointReader(const std::string& filename,
                                   TF_Status* status)
    : reader_(nullptr),
      v2_reader_(nullptr),
      var_to_shape_map_(nullptr),
      var_to_data_type_map_(nullptr) {
  // Depending on whether this is a V2 ckpt, initializes "reader_" or
  // "v2_reader_".
  std::vector<std::string> v2_path;
  if (tf::Env::Default()
          ->GetMatchingPaths(tf::MetaFilename(filename), &v2_path)
          .ok() &&
      !v2_path.empty()) {
    v2_reader_.reset(
        new tf::BundleReader(tf::Env::Default(), filename));
    if (!v2_reader_->status().ok()) {
      tsl::Set_TF_Status_from_Status(status, v2_reader_->status());
      return;
    }
    auto result = BuildV2VarMaps();
    var_to_shape_map_.swap(result.first);
    var_to_data_type_map_.swap(result.second);
  } else {
    reader_.reset(new tf_ckpt::TensorSliceReader(filename));
    if (!reader_->status().ok()) {
      tsl::Set_TF_Status_from_Status(status, reader_->status());
      return;
    }
    var_to_shape_map_.reset(new tf_ckpt::TensorSliceReader::VarToShapeMap(
        reader_->GetVariableToShapeMap()));
    var_to_data_type_map_.reset(
        new tf_ckpt::TensorSliceReader::VarToDataTypeMap(
            reader_->GetVariableToDataTypeMap()));
  }
}

bool CheckpointReader::HasTensor(const std::string& name) const {
  if (reader_ != nullptr) {
    return reader_->HasTensor(name, nullptr, nullptr);
  }
  return v2_reader_->Contains(name);
}

const tf_ckpt::TensorSliceReader::VarToShapeMap&
CheckpointReader::GetVariableToShapeMap() const {
  CHECK(var_to_shape_map_);
  return *var_to_shape_map_;
}

const tf_ckpt::TensorSliceReader::VarToDataTypeMap&
CheckpointReader::GetVariableToDataTypeMap() const {
  CHECK(var_to_data_type_map_);
  return *var_to_data_type_map_;
}

const std::string CheckpointReader::DebugString() const {
  if (reader_ != nullptr) return reader_->DebugString();
  return v2_reader_->DebugString();
}

void CheckpointReader::GetTensor(
    const std::string& name, std::unique_ptr<tf::Tensor>* out_tensor,
    TF_Status* out_status) const {
  absl::Status status;
  if (reader_ != nullptr) {
    status = reader_->GetTensor(name, out_tensor);
  } else {
    tf::DataType dtype;
    tf::TensorShape shape;
    status = v2_reader_->LookupDtypeAndShape(name, &dtype, &shape);
    if (status.ok()) {
      out_tensor->reset(new tf::Tensor(dtype, shape));
      status = v2_reader_->Lookup(name, out_tensor->get());
      if (!status.ok()) out_tensor->reset();
    }
  }
  if (!status.ok()) {
    tsl::Set_TF_Status_from_Status(out_status, status);
  }
}

std::pair<std::unique_ptr<tf_ckpt::TensorSliceReader::VarToShapeMap>,
          std::unique_ptr<tf_ckpt::TensorSliceReader::VarToDataTypeMap>>
CheckpointReader::BuildV2VarMaps() {
  CHECK(v2_reader_ != nullptr);
  CHECK(v2_reader_->status().ok());

  // First pass: filters out the entries of the slices.
  absl::flat_hash_set<std::string> filtered_keys;
  tf::BundleEntryProto entry;
  v2_reader_->Seek(tf::kHeaderEntryKey);
  for (v2_reader_->Next(); v2_reader_->Valid(); v2_reader_->Next()) {
    CHECK(entry.ParseFromString(v2_reader_->value()))
        << entry.InitializationErrorString();
    for (int i = 0; i < entry.slices_size(); ++i) {
      const auto& slice_proto = entry.slices(i);
      CHECK(filtered_keys
                .insert(tf_ckpt::EncodeTensorNameSlice(
                    std::string(v2_reader_->key()) /* full var's name */,
                    tf::TensorSlice(slice_proto)))
                .second);
    }
  }

  // Second pass: adds the entries, ignoring the filtered keys.
  std::unique_ptr<tf_ckpt::TensorSliceReader::VarToShapeMap> var_to_shape_map(
      new tf_ckpt::TensorSliceReader::VarToShapeMap);
  std::unique_ptr<tf_ckpt::TensorSliceReader::VarToDataTypeMap>
      var_to_data_type_map(new tf_ckpt::TensorSliceReader::VarToDataTypeMap);
  v2_reader_->Seek(tf::kHeaderEntryKey);
  for (v2_reader_->Next(); v2_reader_->Valid(); v2_reader_->Next()) {
    if (filtered_keys.count(std::string(v2_reader_->key())) > 0) continue;
    CHECK(entry.ParseFromString(v2_reader_->value()))
        << entry.InitializationErrorString();
    std::string key(v2_reader_->key());
    (*var_to_shape_map)[key] = tf::TensorShape(entry.shape());
    (*var_to_data_type_map)[key] = tf::DataType(entry.dtype());
  }
  // The returned pointers are owned by the caller.
  return std::make_pair(std::move(var_to_shape_map),
                        std::move(var_to_data_type_map));
}

}  // namespace checkpoint
}  // namespace aggregation
}  // namespace tensorflow_federated

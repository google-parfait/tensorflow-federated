/*
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TENSORFLOW_CHECKPOINT_READER_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TENSORFLOW_CHECKPOINT_READER_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"

namespace tensorflow_federated::aggregation::tensorflow {

// This class wraps Tensorflow checkpoint reader and provides a similar
// functionality but returns Aggregation Core tensors instead.
// This class is designed to read only dense tensors that consist of a
// single slice.
class CheckpointReader final {
 public:
  // CheckpointReader is neither copyable nor moveable
  CheckpointReader(const CheckpointReader&) = delete;
  CheckpointReader& operator=(const CheckpointReader&) = delete;

  using DataTypeMap = absl::flat_hash_map<std::string, DataType>;
  using TensorShapeMap = absl::flat_hash_map<std::string, TensorShape>;

  static absl::StatusOr<std::unique_ptr<CheckpointReader>> Create(
      const std::string& filename);

  const DataTypeMap& GetDataTypeMap() const { return data_type_map_; }
  const TensorShapeMap& GetTensorShapeMap() const { return shape_map_; }

  absl::StatusOr<Tensor> GetTensor(const std::string& name) const;

 private:
  CheckpointReader(std::unique_ptr<::tensorflow::checkpoint::CheckpointReader>
                       tensorflow_checkpoint_reader,
                   DataTypeMap data_type_map, TensorShapeMap shape_map);

  std::unique_ptr<::tensorflow::checkpoint::CheckpointReader>
      tf_checkpoint_reader_;
  DataTypeMap data_type_map_;
  TensorShapeMap shape_map_;
};

}  // namespace tensorflow_federated::aggregation::tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_TENSORFLOW_CHECKPOINT_READER_H_

/*
 * Copyright 2024 Google LLC
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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"

#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

constexpr static int64_t kNumTensors = 1000;
constexpr static int64_t kTensorLength = 1000;

template <typename T>
std::unique_ptr<TensorData> CreateRandomTestData() {
  auto test_data = std::make_unique<MutableVectorData<T>>(kTensorLength);
  std::generate(test_data->begin(), test_data->end(),
                []() { return std::rand() % 10000; });
  return test_data;
}

template <>
std::unique_ptr<TensorData> CreateRandomTestData<string_view>() {
  auto test_data = std::make_unique<MutableStringData>(kTensorLength);
  for (int i = 0; i < kTensorLength; ++i) {
    // Add random strings with lengths ranging from 21 to 25 bytes.
    test_data->Add(std::string(20, 'x') + std::to_string(std::rand() % 10000));
  }
  return test_data;
}

std::vector<Tensor> CreateRandomTensors(DataType dtype) {
  std::vector<Tensor> tensors(kNumTensors);
  for (int i = 0; i < kNumTensors; ++i) {
    DTYPE_CASES(
        dtype, T,
        tensors[i] = std::move(Tensor::Create(dtype, {kTensorLength},
                                              CreateRandomTestData<T>()))
                         .value());
  }
  return tensors;
}

static void BM_CombineKeys(benchmark::State& state) {
  int64_t num_input_tensors = state.range(0);
  DataType dtype = static_cast<DataType>(state.range(1));
  std::vector<Tensor> tensors = CreateRandomTensors(dtype);

  std::vector<DataType> dtypes(num_input_tensors);
  InputTensorList input_list(num_input_tensors);

  for (int i = 0; i < num_input_tensors; ++i) {
    dtypes[i] = dtype;
  }

  // Create a CompositeKeyCombiner with the given data type.
  CompositeKeyCombiner combiner(dtypes);
  auto items_processed = 0;
  for (auto s : state) {
    // Pick num_input_tensors random tensors.
    for (int i = 0; i < num_input_tensors; ++i) {
      input_list[i] = &tensors[std::rand() % tensors.size()];
    }
    benchmark::DoNotOptimize(combiner.Accumulate(input_list));
    items_processed += kTensorLength;
  }
  state.SetItemsProcessed(items_processed);
}

BENCHMARK(BM_CombineKeys)
    ->Args({1, DT_INT32})
    ->Args({4, DT_INT32})
    ->Args({16, DT_INT32})
    ->Args({1, DT_INT64})
    ->Args({4, DT_INT64})
    ->Args({16, DT_INT64})
    ->Args({1, DT_STRING})
    ->Args({4, DT_STRING})
    ->Args({16, DT_STRING});

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated

// Run the benchmark
BENCHMARK_MAIN();

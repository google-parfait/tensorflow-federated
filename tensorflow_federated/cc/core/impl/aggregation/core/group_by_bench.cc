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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/benchmark/benchmark.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_registry.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

constexpr static int64_t kTensorLength = 1000000;

static void BM_GroupBySumAccumulate(benchmark::State& state) {
  Intrinsic inner_intrinsic = Intrinsic{"GoogleSQL:sum",
                                        {TensorSpec("value", DT_FLOAT, {-1})},
                                        {TensorSpec("value", DT_FLOAT, {-1})},
                                        {},
                                        {}};
  Intrinsic intrinsic{
      "fedsql_group_by",
      {TensorSpec("key1", DT_INT32, {-1}), TensorSpec("key2", DT_STRING, {-1})},
      {TensorSpec("key1_out", DT_INT32, {-1}),
       TensorSpec("key2_out", DT_STRING, {-1})},
      {},
      {}};
  intrinsic.nested_intrinsics.push_back(std::move(inner_intrinsic));
  std::unique_ptr<TensorAggregator> aggregator =
      CreateTensorAggregator(intrinsic).value();

  std::vector<std::string> string_options = {"cat", "dog", "bird", "fish",
                                             "mouse"};
  auto int_keys = std::make_unique<MutableVectorData<float>>(kTensorLength);
  std::vector<float>& int_keys_vec = *int_keys;
  auto string_keys = std::make_unique<MutableStringData>(kTensorLength);
  auto values = std::make_unique<MutableVectorData<float>>(kTensorLength);
  std::vector<float> values_vec = *values;
  for (int64_t i = 0; i < kTensorLength; ++i) {
    int_keys_vec[i] = i % 123;
    std::string string_key = string_options[i % string_options.size()];
    string_keys->Add(std::move(string_key));
    values_vec[i] = (i % 123) * .1;
  }
  auto int_keys_tensor =
      Tensor::Create(DT_INT32, {kTensorLength}, std::move(int_keys)).value();
  auto string_keys_tensor =
      Tensor::Create(DT_STRING, {kTensorLength}, std::move(string_keys))
          .value();
  auto values_tensor =
      Tensor::Create(DT_FLOAT, {kTensorLength}, std::move(values)).value();
  auto items_processed = 0;

  // Benchmark time is only measured in the loop body.
  for (auto s : state) {
    benchmark::DoNotOptimize(aggregator->Accumulate(
        {&int_keys_tensor, &string_keys_tensor, &values_tensor}));
    items_processed += kTensorLength;
  }
  state.SetItemsProcessed(items_processed);
}

BENCHMARK(BM_GroupBySumAccumulate);

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated

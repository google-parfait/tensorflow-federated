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

#ifndef TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_QUANTILE_AGGREGATOR_H_
#define TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_QUANTILE_AGGREGATOR_H_

#include <cmath>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"

namespace tensorflow_federated {
namespace aggregation {
// Wrapping around std::ceil to return an int.
int int_ceil(double val);

// DPQuantileAggregator ingests a stream of input scalars of type T.
// It stores the inputs in a buffer, using reservoir sampling if necessary.
// When ReportWithEpsilonAndDelta() is called, it will employ (a variant of) the
// algorithm by Durfee to find the quantile subject to differential privacy.
// https://proceedings.neurips.cc/paper_files/paper/2023/file/f4b6ef2a78684dca2fb3f1c09372e041-Paper-Conference.pdf
template <typename T>
class DPQuantileAggregator final : public DPTensorAggregator {
 public:
  explicit DPQuantileAggregator(double target_quantile, int num_inputs,
                                int reservoir_sampling_count,
                                std::unique_ptr<MutableVectorData<T>> buffer)
      : DPTensorAggregator(),
        target_quantile_(target_quantile),
        num_inputs_(num_inputs),
        reservoir_sampling_count_(reservoir_sampling_count),
        buffer_(*std::move(buffer)),
        output_consumed_(false) {
    TFF_CHECK(target_quantile > 0 && target_quantile < 1)
        << "Target quantile must be in (0, 1).";
  }
  explicit DPQuantileAggregator(double target_quantile)
      : DPQuantileAggregator(target_quantile, 0, 0,
                             std::make_unique<MutableVectorData<T>>()) {}

  inline int GetNumInputs() const override { return num_inputs_; }

  inline int GetBufferSize() const { return buffer_.size(); }

  inline int GetReservoirSamplingCount() const {
    return reservoir_sampling_count_;
  }

  // To MergeWith another DPQuantileAggregator, we will insert as many elements
  // from the other aggregator's buffer as possible into our buffer, without
  // exceeding kDPQuantileMaxInputs. If there are remaining elements in the
  // other buffer, we will perform reservoir sampling.
  Status MergeWith(TensorAggregator&& other) override;

  StatusOr<std::string> Serialize() && override {
    DPQuantileAggregatorState aggregator_state;
    aggregator_state.set_num_inputs(num_inputs_);
    aggregator_state.set_reservoir_sampling_count(reservoir_sampling_count_);
    *(aggregator_state.mutable_buffer()) = buffer_.EncodeContent();
    return aggregator_state.SerializeAsString();
  }

  // Trigger execution of the DP quantile algorithm from Durfee's paper. It is
  // an application of the textbook AboveThreshold algorithm to prefix sums:
  // this algorithm will loop through buckets and privately estimate how much
  // of buffer_ belongs to the buckets scanned so far.
  // If the estimate exceeds a noisy version of
  // (target_quantile_ * buffer_.size()), the algorithm returns the bucket.
  StatusOr<OutputTensorList>
      ReportWithEpsilonAndDelta(double epsilon, double delta) && override;

  // Given a value, return the bucket that it belongs to. Buckets are partitions
  // of the number line, indexed from 0. The bucket boundaries are at first
  // linear (governed by kDPQuantileLinearRate), then exponential (governed by
  // kDPQuantileExponentialRate); the error due to bucketing is additively
  // kDPQuantileLinearRate and multiplicatively kDPQuantileExponentialRate.
  inline int GetBucket(double value) const {
    if (value < 0) {
      return 0;
    } else if (value < kDPQuantileEndOfLinearGrowth) {
      // Find the smallest multiplier of kDPQuantileLinearRate that results in
      // a value greater than or equal to the input.
      return int_ceil(value / kDPQuantileLinearRate);
    } else {
      // Find the smallest integer exponent such that
      // kDPQuantileEndOfLinearGrowth * (kDPQuantileExponentialRate^exponent)
      // >= value.
      int exponent = int_ceil(std::log(value / kDPQuantileEndOfLinearGrowth) /
                              std::log(kDPQuantileExponentialRate));
      return exponent +
             int_ceil(kDPQuantileEndOfLinearGrowth / kDPQuantileLinearRate);
    }
  }

  // A bucket corresponds to a range of values; calculate its upper bound.
  inline double BucketUpperBound(int bucket) const {
    double candidate = bucket * kDPQuantileLinearRate;
    if (candidate < kDPQuantileEndOfLinearGrowth) {
      return candidate;
    }
    int exponent =
        bucket - int_ceil(kDPQuantileEndOfLinearGrowth / kDPQuantileLinearRate);
    return kDPQuantileEndOfLinearGrowth *
           std::pow(kDPQuantileExponentialRate, exponent);
  }

  // Calculate the rank of the target quantile in the buffer.
  inline double GetTargetRank() const {
    return target_quantile_ * static_cast<double>(buffer_.size());
  }

 protected:
  // This DP mechanism expects one scalar tensor in the input. It pushes the
  // scalar into the buffer if the buffer is smaller than kDPQuantileMaxInputs.
  // Otherwise, it will perform reservoir sampling
  Status AggregateTensors(InputTensorList tensors) override;

  // Checks if the output has not already been consumed.
  Status CheckValid() const override;

 private:
  // Implements Vitter's reservoir sampling algorithm.
  // https://en.wikipedia.org/wiki/Reservoir_sampling#Simple:_Algorithm_R
  // Called by AggregateTensors and MergeWith when buffer_ is full.
  inline void InsertWithReservoirSampling(T value);

  // PrefixSumAboveThreshold iterates over histogram buckets and privately
  // estimates prefix sums. If the noisy prefix sum exceeds a noisy threshold,
  // it returns the bucket.
  // This algorithm ensures epsilon-DP when each client contributes exactly one
  // value to exactly one bucket of the histogram.
  StatusOr<int> PrefixSumAboveThreshold(
      double epsilon, absl::flat_hash_map<int, int>& histogram,
      double threshold, int max_bucket);

  double target_quantile_;
  int num_inputs_, reservoir_sampling_count_;
  MutableVectorData<T> buffer_;
  absl::BitGen bit_gen_;
  bool output_consumed_;
};

// This factory class makes DPQuantileAggregator, defined in the cc file.
// It expects only one parameter in the input intrinsic: the target quantile.
class DPQuantileAggregatorFactory final : public TensorAggregatorFactory {
 public:
  DPQuantileAggregatorFactory() = default;

  // DPQuantileAggregatorFactory isn't copyable or moveable.
  DPQuantileAggregatorFactory(const DPQuantileAggregatorFactory&) = delete;
  DPQuantileAggregatorFactory& operator=(const DPQuantileAggregatorFactory&) =
      delete;

  StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override {
    DPQuantileAggregatorState aggregator_state;
    if (!aggregator_state.ParseFromString(serialized_state)) {
      return TFF_STATUS(INVALID_ARGUMENT)
             << "DPQuantileAggregatorFactory: Failed to parse serialized "
                "state.";
    }
    return CreateInternal(intrinsic, &aggregator_state);
  }

  StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override {
    return CreateInternal(intrinsic, nullptr);
  }

 private:
  StatusOr<std::unique_ptr<TensorAggregator>> CreateInternal(
      const Intrinsic& intrinsic,
      const DPQuantileAggregatorState* aggregator_state) const;
};
}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DP_QUANTILE_AGGREGATOR_H_

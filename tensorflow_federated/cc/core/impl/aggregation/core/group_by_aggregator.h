/*
 * Copyright 2023 Google LLC
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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_GROUP_BY_AGGREGATOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_GROUP_BY_AGGREGATOR_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/agg_core.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/input_tensor_list.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/one_dim_grouping_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_aggregator_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"

namespace tensorflow_federated {
namespace aggregation {

// GroupByAggregator class is a specialization of TensorAggregator which
// takes in a predefined number of tensors to be used as keys and and predefined
// number of tensors to be used as values. It computes the unique combined keys
// across all tensors and then accumulates the values into the output positions
// matching those of the corresponding keys.
//
// Currently only 1D input tensors are supported.
//
// The specific means of accumulating values are delegated to an inner
// OneDimGroupingAggregator intrinsic for each value tensor that should be
// grouped.
//
// When Report is called, this TensorAggregator outputs all key types for which
// the output_key_specs have a nonempty tensor name, as well as all value
// tensors output by the OneDimGroupingAggregator intrinsics.
//
// This class is not thread safe.
class GroupByAggregator : public TensorAggregator {
 public:
  // Merge this GroupByAggregator with another GroupByAggregator that operates
  // on compatible types using compatible inner intrinsics.
  Status MergeWith(TensorAggregator&& other) override;

  // Returns the number of inputs that have been accumulated or merged into this
  // GroupByAggregator.
  int GetNumInputs() const override { return num_inputs_; }

  // Returns the number of contributors that have been accumulated or merged
  // into this GroupByAggregator.
  const std::vector<int>& GetContributors() const {
    return contributors_to_groups_;
  }

  // Override CanReport to ensure that outputs of Report will contain all
  // expected output tensors. It is not valid to create empty tensors, so in
  // order to produce a report containing the expected number of output tensors,
  // at least one input must have been aggregated.
  bool CanReport() const override;

 protected:
  friend class GroupByFactory;

  // Constructs a GroupByAggregator.
  //
  // This constructor is meant for use by the GroupByFactory; most callers
  // should instead create a GroupByAggregator from an intrinsic using the
  // factory, i.e.
  // `(*GetAggregatorFactory("fedsql_group_by"))->Create(intrinsic)`
  //
  //
  // Takes in the following inputs:
  //
  // input_key_specs: A vector of TensorSpecs for the tensors that this
  // GroupByAggregator should treat as keys in the input. The first n tensors in
  // any InputTensorList provided to an Accumulate call are expected to match
  // the n TensorSpecs in this vector. For now the shape of each tensor should
  // be {-1} as only one-dimensional aggregations are supported and different
  // calls to Accumulate may have different numbers of examples in each tensor.
  //
  // output_key_specs: A vector of TensorSpecs providing this GroupByAggregator
  // with information on which key tensors should be included in the output.
  // An empty string for the tensor name indicates that the tensor should not
  // be included in the output.
  // Regardless of the output_key_specs, all key tensors listed in the
  // input_key_specs will be used for grouping.
  // output_key_specs must have the same number of TensorSpecs as
  // input_key_specs, and all TensorSpec attributes but the tensor names must
  // match those in input_key_specs. The lifetime of output_key_specs must
  // outlast this class.
  //
  // intrinsics: Pointer to a vector of Intrinsic classes that should contain
  // subclasses of OneDimGroupingAggregator to which this class will delegate
  // grouping of values.
  // The number of tensors in each InputTensorList provided to Accumulate must
  // match the number of TensorSpecs in input_key_specs plus the number of
  // Intrinsics in this vector.
  //
  // key_combiner: either nullptr or a smart pointer to a CompositeKeyCombiner.
  //
  // aggregators: a vector of unique_ptrs to TensorAggregators made by the
  // factory. Used to perform the inner aggregations.
  //
  // num_inputs: the number of inputs initially represented by the aggregator.
  //
  // min_contributors_to_group: the minimum number of contributors before a
  // group total can be released (i.e. the threshold for k-thresholding). If not
  // set, no check will be made on the number of contributors to a group before
  // release.
  //
  // max_contributors_to_group: the maximum number of contributors to a group
  // to keep track of. Must not be set if min_contributors_to_group is not set.
  // If not set (but min_contributors_to_group is set) then will be set to the
  // same value as min_contributors_to_group.
  //
  // This class takes ownership of the intrinsics vector and the aggregators
  // vector.
  GroupByAggregator(
      const std::vector<TensorSpec>& input_key_specs,
      const std::vector<TensorSpec>* output_key_specs,
      const std::vector<Intrinsic>* intrinsics,
      std::unique_ptr<CompositeKeyCombiner> key_combiner,
      std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators,
      int num_inputs,
      std::optional<int> min_contributors_to_group = std::nullopt);

  // Creates a vector of DataTypes that describe the keys in the input & output.
  // A pre-processing function that sets the stage for CompositeKeyCombiners.
  static std::vector<DataType> CreateKeyTypes(
      size_t num_keys_per_input, const std::vector<TensorSpec>& input_key_specs,
      const std::vector<TensorSpec>& output_key_specs);

  // Perform aggregation of the tensors in a single InputTensorList into the
  // state of this GroupByAggregator and increment the count of aggregated
  // tensors.
  //
  // The order in which the tensors must appear in the input is the following:
  // first, the key tensors in the order they appear in the input_tensor_specs,
  // and next, the value tensors in the order the inner intrinsics that
  // aggregate each value appear in the intrinsics input vector.
  Status AggregateTensors(InputTensorList tensors) override;

  // Ensures that the output has not yet been consumed for this
  // GroupByAggregator.
  Status CheckValid() const override;

  // Produce final outputs from this GroupByAggregator. Keys will only be output
  // for those tensors with nonempty tensor names in the output_key_specs_.
  // Values will be output from all inner intrinsics.
  //
  // The order in which the tensors will appear in the output is the following:
  // first, the keys with nonempty tensor names in the order they appear in the
  // output_tensor_specs, and next, the value tensors in the order the inner
  // intrinsics that produce each value tensor appear in the intrinsics input
  // vector.
  //
  // Once this function is called, CheckValid will return false.
  OutputTensorList TakeOutputs() && override;

  // The virtual function below enables a distinction between creating ordinals
  // within MergeTensorsInternal and within AggregateTensorsInternal.
  // Refer to CreateOrdinalsByGroupingKeys for the latter.
  virtual StatusOr<Tensor> CreateOrdinalsByGroupingKeysForMerge(
      const InputTensorList& inputs);

  StatusOr<std::string> Serialize() && override;

  inline size_t num_keys_per_input() const { return num_keys_per_input_; }
  inline std::unique_ptr<CompositeKeyCombiner>& key_combiner() {
    return key_combiner_;
  }
  inline const std::vector<Intrinsic>& intrinsics() const {
    return intrinsics_;
  }
  inline const std::vector<TensorSpec>& output_key_specs() const {
    return output_key_specs_;
  }

 private:
  // Returns either nullptr or a unique_ptr to a CompositeKeyCombiner, depending
  // on the input specification. Relies on CreateKeyTypes.
  static std::unique_ptr<CompositeKeyCombiner> CreateKeyCombiner(
      const std::vector<TensorSpec>& input_key_specs,
      const std::vector<TensorSpec>* output_key_specs);

  // Checks that the input tensor at the provided index has the expected type
  // and shape.
  static Status ValidateInputTensor(const InputTensorList& tensors,
                                    size_t input_index,
                                    const TensorSpec& expected_tensor_spec,
                                    const TensorShape& key_shape);

  // Internal implementation to accumulate the input tensors into the state of
  // this GroupByAggregator.
  Status AggregateTensorsInternal(InputTensorList tensors);

  // Internal implementation to merge the input tensors into the state of this
  // GroupByAggregator. The num_merged_inputs arg contains the number of inputs
  // that were pre-accumulated into the tensors input param.
  Status MergeTensorsInternal(InputTensorList tensors, int num_merged_inputs);

  // Internal implementation of TakeOutputs that returns all keys and values,
  // including keys that should not actually be returned in the final output.
  // Once this function is called, CheckValid will return false.
  OutputTensorList TakeOutputsInternal();

  // If there are key tensors for this GroupByAggregator, then group key inputs
  // into unique composite keys, and produce an ordinal for each element of the
  // input corresponding to the index of the unique composite key in the output.
  // Otherwise, produce an ordinals vector of the same shape as the inputs, but
  // made up of all zeroes, so that all elements will be aggregated into a
  // single output element.
  StatusOr<Tensor> CreateOrdinalsByGroupingKeys(const InputTensorList& inputs);

  // Returns OK if the input and output tensor specs of the intrinsics
  // held by other match those of the sub-intrinsics held by this
  // GroupByAggregator, and the data types of input keys and the TensorSpecs of
  // output keys match those for this GroupByAggregator. Otherwise returns
  // INVALID_ARGUMENT.
  // TODO: b/280653641 - Also validate that intrinsic URIs match.
  Status IsCompatible(const GroupByAggregator& other) const;

  bool output_consumed_ = false;
  int num_inputs_;
  const size_t num_keys_per_input_;
  size_t num_tensors_per_input_;
  std::unique_ptr<CompositeKeyCombiner> key_combiner_;
  const std::vector<Intrinsic>& intrinsics_;
  const std::vector<TensorSpec>& output_key_specs_;
  std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>> aggregators_;
  std::optional<int> min_contributors_to_group_;
  std::optional<int> max_contributors_to_group_;
  std::vector<int> contributors_to_groups_;
};

// Factory class for the GroupByAggregator.
class GroupByFactory final : public TensorAggregatorFactory {
 public:
  GroupByFactory() = default;

  // GroupByFactory isn't copyable or moveable.
  GroupByFactory(const GroupByFactory&) = delete;
  GroupByFactory& operator=(const GroupByFactory&) = delete;

  StatusOr<std::unique_ptr<TensorAggregator>> Create(
      const Intrinsic& intrinsic) const override;

  StatusOr<std::unique_ptr<TensorAggregator>> Deserialize(
      const Intrinsic& intrinsic, std::string serialized_state) const override;

  // Check that the configuration is valid for SQL grouping aggregators.
  static Status CheckIntrinsic(const Intrinsic& intrinsic, const char* uri);

  // Create a vector of inner OneDimBaseGroupingAggregators. If state is
  // provided, the inner aggregators will be constructed using their portion of
  // the state.
  static StatusOr<std::vector<std::unique_ptr<OneDimBaseGroupingAggregator>>>
  CreateAggregators(const Intrinsic& intrinsic,
                    const GroupByAggregatorState* aggregator_state);

  // Adds keys from the aggregator state, if any, to the composite key combiner.
  static Status PopulateKeyCombinerFromState(
      CompositeKeyCombiner& key_combiner,
      const GroupByAggregatorState& aggregator_state);

 private:
  // Create a GroupByAggregator. If state is provided, the GroupByAggregator
  // will be constructed using the state.
  StatusOr<std::unique_ptr<TensorAggregator>> CreateInternal(
      const Intrinsic& intrinsic,
      const GroupByAggregatorState* aggregator_state) const;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_GROUP_BY_AGGREGATOR_H_

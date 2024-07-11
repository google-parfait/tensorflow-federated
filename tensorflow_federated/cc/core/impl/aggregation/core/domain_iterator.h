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

#ifndef TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DOMAIN_ITERATOR_H_
#define TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DOMAIN_ITERATOR_H_

#include <cstdint>

#include "absl/container/fixed_array.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_composite_key_combiner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace tensorflow_federated {
namespace aggregation {

namespace internal {

// A DomainIterator allows DPClosedDomainHistogram::Report() to loop through all
// possible combinations of grouping keys (a.k.a. composite keys) in a data-
// independent order. The domain of key i is described by Tensor i of the member
// domain_tensors_.
// If the domain of key 0 is (a, b, c) and the domain of key 1 is (d ,e), then
// this class proceeds through (a, d), (b, d), (c, d), (a, e), (b, e), (c, e).
//
// operator*() returns a pointer to some value of interest; child classes define
// what that is.
class DomainIterator {
 public:
  // This class must be initialized with the domains of the keys.
  DomainIterator() = delete;
  explicit DomainIterator(TensorSpan domain_tensors)
      : domain_tensors_(domain_tensors),
        domain_indices_(domain_tensors.size(), 0),
        wrapped_around_(false) {}

  virtual ~DomainIterator() = default;

  // Returns a pointer to some value of interest.
  virtual const void* operator*() = 0;

  // Returns true if we wrapped around the end of the composite domain.
  bool wrapped_around() const { return wrapped_around_; }

 protected:
  // Advances the domain_indices_ such that they specify the next composite key
  // in the domain. Performs modular wraparound if necessary. Updates
  // wrapped_around_ if we reach the end of the domain.
  void Increment(int64_t which_key = 0);

  TensorSpan domain_tensors() const { return domain_tensors_; };
  absl::FixedArray<int64_t>& domain_indices() { return domain_indices_; }

 private:
  // The specification of the domain of each key.
  TensorSpan domain_tensors_;

  // The indices of the current composite key in the domain. For the example
  // above, domain_indices_ = (2, 0) encodes (c, d).
  absl::FixedArray<int64_t> domain_indices_;

  bool wrapped_around_;
};

// DPClosedDomainHistogram::Report() will need to produce, for each key intended
// to be output, a Tensor of possible values of that key. This child of
// DomainIterator supports that functionality.
// The which_key_ member specifies the index of the key in domain_tensors_.
// If which_key_ = 0, then operator*() returns a pointer to the i-th member of
// the domain of key 0, where i = domain_indices_[0] is set by Increment().
// Continuing with the preceding example, calling operator*() after every
// operator++() call will yield pointers to a, b, c, a, b, c.
// If which_key_ = 1, then the pointers move through d, d, d, e, e, e.
class DomainIteratorForKeys : public DomainIterator {
 public:
  DomainIteratorForKeys(TensorSpan domain_tensors, int64_t which_key)
      : DomainIterator(domain_tensors), which_key_(which_key) {}

  // Public interface to Increment()
  DomainIteratorForKeys& operator++();

  const void* operator*() override;

 private:
  int64_t which_key_;
};

// DPClosedDomainHistogram::Report() will need to produce, for each composite
// key, the aggregations corresponding to that key. This child of DomainIterator
// supports that functionality.
// It is initialized with a Tensor of aggregates and a DPCompositeKeyCombiner.
// If the current state of domain_indices_ encodes (b, e), then operator*() will
// return a pointer to the aggregate corresponding to (b, e).
// Specifically, if (b, e) was assigned the ordinal 2 by the member
// dp_key_combiner_ and aggregation_ contains (3, 8, 10, 12), then the output is
// a pointer to the 10 in the Tensor.
class DomainIteratorForAggregations : public DomainIterator {
 public:
  DomainIteratorForAggregations(TensorSpan domain_tensors,
                                const Tensor& aggregation,
                                DPCompositeKeyCombiner& dp_key_combiner)
      : DomainIterator(domain_tensors),
        aggregation_(aggregation),
        dp_key_combiner_(dp_key_combiner) {}

  // Public interface to Increment()
  DomainIteratorForAggregations& operator++();

  const void* operator*() override;

 private:
  const Tensor& aggregation_;
  DPCompositeKeyCombiner& dp_key_combiner_;
};

}  // namespace internal
}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_DOMAIN_ITERATOR_H_

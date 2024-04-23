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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_AGGREGATOR_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_AGGREGATOR_H_

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"

namespace tensorflow_federated {
namespace aggregation {

// Abstract base for aggregators that compute an aggregate of input items of
// type T into a final aggregate of type R using a multi-stage process in which
// items are first partially aggregated at an intermediate layer, then the
// partial aggregates are further combined, and finally projected into the
// result. This multi-stage process consists of the following:
// a) The aggregator is created with a zero value of an arbitrary intermediate
//    type U. Please note that the type U is never surfaced and considered an
//    implementation detail, so it doesn't need to be explicitlty parameterized.
// b) The method Accumulate is used to accumulate T-typed client items into the
//    U-typed partial aggregate.
// c) The method Merge is used to merge the intermediate U-typed aggregates of
//    the two aggregator instances producing a merged U-typed aggregate.
// d) The method Report is used to project the top-level U-typed aggregate into
//    the final R-typed result.
// The typename Self is used to specify the actual derived class.
template <typename T, typename R, typename Self>
class Aggregator {
 public:
  Aggregator() = default;
  virtual ~Aggregator() = default;

  // Aggregator derived classes are not copyable.
  Aggregator(const Aggregator&) = delete;

  // Accumulates an input into the intermediate aggregate.
  // The method may fail if the input isn't compatible with the current
  // Aggregator or if the Aggregator instance has already been 'consumed'.
  virtual Status Accumulate(T input) = 0;

  // Merges intermediate aggregates from the other Aggregator instance into the
  // current Aggregator instance. Doing so 'consumes' the other Aggregator
  // instance.
  // The method may fail if the two Aggregator instances aren't compatible.
  virtual Status MergeWith(Self&& other) = 0;

  // Returns true if the current Aggregator instance can produce a report, for
  // example if a sufficient number of inputs has been accumulated.
  virtual bool CanReport() const = 0;

  // Produces the final report, 'consuming' the current Aggregator instance.
  // Once the current instance is consumed it can no longer perform any
  // operations.
  // This method fails when CanReport method returns false.
  virtual StatusOr<R> Report() && = 0;
};

}  // namespace aggregation
}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_CORE_AGGREGATOR_H_

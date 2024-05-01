/*
 * Copyright 2019 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/base/unique_value.h"

#include <utility>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"

namespace tensorflow_federated {

using ::testing::Eq;

struct ValueBox {
  bool destructed = false;
  int value = 0;
};

class TracedValue {
 public:
  explicit TracedValue(int value) : local_value_(0), box_(nullptr) {
    UpdateValue(value);
  }

  void AttachToBox(ValueBox* box) {
    TFF_CHECK(box_ == nullptr);
    box_ = box;
    UpdateValue(local_value_);
  }

  TracedValue(TracedValue const& other) : local_value_(0), box_(nullptr) {
    UpdateValue(other.value());
  }

  TracedValue& operator=(TracedValue const& other) {
    UpdateValue(other.value());
    return *this;
  }

  ~TracedValue() {
    if (box_) {
      box_->destructed = true;
    }
  }

  int value() const { return local_value_; }

 private:
  void UpdateValue(int value) {
    local_value_ = value;
    if (box_) {
      box_->destructed = false;
      box_->value = value;
    }
  }

  int local_value_;
  ValueBox* box_;
};

TEST(UniqueValueTest, MoveToInnerScope) {
  ValueBox box_a{};
  ValueBox box_b{};

  {
    UniqueValue<TracedValue> a = MakeUniqueValue<TracedValue>(123);
    a->AttachToBox(&box_a);
    EXPECT_THAT(box_a.destructed, Eq(false));
    EXPECT_THAT(box_a.value, Eq(123));

    {
      UniqueValue<TracedValue> b = MakeUniqueValue<TracedValue>(456);
      b->AttachToBox(&box_b);
      EXPECT_THAT(box_b.destructed, Eq(false));
      EXPECT_THAT(box_b.value, Eq(456));

      b = std::move(a);

      EXPECT_THAT(box_a.destructed, Eq(true));
      EXPECT_THAT(box_b.destructed, Eq(false));
      EXPECT_THAT(box_b.value, Eq(123));
    }

    EXPECT_THAT(box_a.destructed, Eq(true));
    EXPECT_THAT(box_b.destructed, Eq(true));
  }
}

TEST(UniqueValueTest, MoveToOuterScope) {
  ValueBox box_a{};
  ValueBox box_b{};

  {
    UniqueValue<TracedValue> a = MakeUniqueValue<TracedValue>(123);
    a->AttachToBox(&box_a);
    EXPECT_THAT(box_a.destructed, Eq(false));
    EXPECT_THAT(box_a.value, Eq(123));

    {
      UniqueValue<TracedValue> b = MakeUniqueValue<TracedValue>(456);
      b->AttachToBox(&box_b);
      EXPECT_THAT(box_b.destructed, Eq(false));
      EXPECT_THAT(box_b.value, Eq(456));

      a = std::move(b);

      EXPECT_THAT(box_a.destructed, Eq(false));
      EXPECT_THAT(box_a.value, Eq(456));
      EXPECT_THAT(box_b.destructed, Eq(true));
    }

    EXPECT_THAT(box_a.destructed, Eq(false));
    EXPECT_THAT(box_a.value, Eq(456));
    EXPECT_THAT(box_b.destructed, Eq(true));
  }

  EXPECT_THAT(box_a.destructed, Eq(true));
  EXPECT_THAT(box_b.destructed, Eq(true));
}

TEST(UniqueValueTest, Emplace) {
  ValueBox box_a{};
  ValueBox box_b{};
  {
    UniqueValue<TracedValue> v{std::nullopt};
    v.Emplace(123);
    v->AttachToBox(&box_a);
    EXPECT_THAT(box_a.destructed, Eq(false));
    EXPECT_THAT(box_a.value, Eq(123));
    v.Emplace(321);
    v->AttachToBox(&box_b);
    EXPECT_THAT(box_a.destructed, Eq(true));
    EXPECT_THAT(box_b.destructed, Eq(false));
    EXPECT_THAT(box_b.value, Eq(321));
  }
}

TEST(UniqueValueTest, Reset) {
  ValueBox box_a{};
  UniqueValue<TracedValue> v = MakeUniqueValue<TracedValue>(123);
  v.Emplace(123);
  v->AttachToBox(&box_a);
  EXPECT_THAT(box_a.destructed, Eq(false));
  EXPECT_THAT(box_a.value, Eq(123));
  v.Reset();
  EXPECT_THAT(box_a.destructed, Eq(true));
  v.Reset();
  EXPECT_THAT(box_a.destructed, Eq(true));
}

}  // namespace tensorflow_federated

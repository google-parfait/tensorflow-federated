/*
 * Copyright 2020 Google LLC
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

#include "tensorflow_federated/cc/core/impl/aggregation/base/simulated_clock.h"

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/time/civil_time.h"
#include "absl/time/time.h"

namespace tensorflow_federated {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;

// Simple callback waiter that runs the function on Wakeup.
class CallbackWaiter : public Clock::Waiter {
 public:
  explicit CallbackWaiter(std::function<void()> callback)
      : callback_(std::move(callback)) {}

  void WakeUp() override { callback_(); }

 private:
  std::function<void()> callback_;
};

// Simple test waiter that adds its ID to the provided vector when WakeUp is
// called. This is used to verify that waiters are woken up in the right order.
class TestWaiter : public CallbackWaiter {
 public:
  explicit TestWaiter(int id, std::vector<int>* output)
      : CallbackWaiter([=]() { output->push_back(id); }) {}
};

absl::Time GetTestInitialTime() {
  return absl::FromCivil(absl::CivilDay(2020, 1, 1), absl::LocalTimeZone());
}

TEST(SimulatedClockTest, GetAndUpdateNow) {
  absl::Time t = absl::UnixEpoch();
  SimulatedClock clock;
  EXPECT_THAT(clock.Now(), t);

  absl::Time t2 = GetTestInitialTime();

  SimulatedClock clock2(t2);
  EXPECT_THAT(clock2.Now(), t2);

  absl::Time t3 = t2 + absl::Seconds(42);
  clock2.AdvanceTime(absl::Seconds(42));
  EXPECT_THAT(clock2.Now(), t3);

  absl::Time t4 = t3 + absl::Seconds(18);
  clock2.SetTime(t4);
  EXPECT_THAT(clock2.Now(), Eq(t4));
}

// Verifies that waiters with future deadlines are not triggered unless the
// time is advanced.
TEST(SimulatedClockTest, FutureDeadline) {
  std::vector<int> output;
  absl::Time t = GetTestInitialTime();
  SimulatedClock clock(t);

  clock.WakeupWithDeadline(t + absl::Seconds(1),
                           std::make_shared<TestWaiter>(1, &output));
  EXPECT_THAT(output, ElementsAre());

  clock.AdvanceTime(absl::Seconds(1));
  EXPECT_THAT(output, ElementsAre(1));

  // Advancing time again doesn't trigger the same waiter again.
  clock.AdvanceTime(absl::Seconds(1));
  EXPECT_THAT(output, ElementsAre(1));
}

// Verifies that the order of waiters with maching deadlines is preserved
// when their wake-up is triggered.
TEST(SimulatedClockTest, MatchingDeadlines) {
  std::vector<int> output;
  absl::Time t = GetTestInitialTime();
  SimulatedClock clock(t);

  absl::Time t1 = t + absl::Seconds(1);
  absl::Time t2 = t + absl::Seconds(2);
  clock.WakeupWithDeadline(t1, std::make_shared<TestWaiter>(1, &output));
  clock.WakeupWithDeadline(t2, std::make_shared<TestWaiter>(2, &output));
  clock.WakeupWithDeadline(t1, std::make_shared<TestWaiter>(3, &output));
  clock.WakeupWithDeadline(t2, std::make_shared<TestWaiter>(4, &output));
  clock.WakeupWithDeadline(t1, std::make_shared<TestWaiter>(5, &output));

  // Trigger all waiters.
  clock.AdvanceTime(absl::Seconds(2));
  EXPECT_THAT(output, ElementsAre(1, 3, 5, 2, 4));
}

// Verifies that waiters with current or past deadlines are triggered promptly.
TEST(SimulatedClockTest, PastAndCurrentDeadlines) {
  std::vector<int> output;
  absl::Time t =
      absl::FromCivil(absl::CivilDay(2020, 1, 1), absl::LocalTimeZone());
  SimulatedClock clock(t);

  clock.WakeupWithDeadline(t, std::make_shared<TestWaiter>(1, &output));
  clock.WakeupWithDeadline(t - absl::Seconds(1),
                           std::make_shared<TestWaiter>(2, &output));
  EXPECT_THAT(output, ElementsAre(1, 2));
}

// Verifies that only expired waiters are triggered.
TEST(SimulatedClockTest, MultipleWaiters) {
  std::vector<int> output;
  absl::Time t = GetTestInitialTime();
  SimulatedClock clock(t);

  clock.WakeupWithDeadline(t + absl::Seconds(30),
                           std::make_shared<TestWaiter>(1, &output));
  clock.WakeupWithDeadline(t + absl::Seconds(20),
                           std::make_shared<TestWaiter>(2, &output));
  clock.WakeupWithDeadline(t + absl::Seconds(10),
                           std::make_shared<TestWaiter>(3, &output));
  // Advance by 15 seconds
  clock.AdvanceTime(absl::Seconds(15));
  // Advance by another 5 seconds
  clock.AdvanceTime(absl::Seconds(5));
  // Only waiters 3 and 2 should be triggered.
  EXPECT_THAT(output, ElementsAre(3, 2));
}

// Verifies that a new timer can be scheduled when anoter timer is triggered.
TEST(SimulatedClockTest, RecursiveWakeup) {
  std::vector<int> output;
  absl::Time t = GetTestInitialTime();
  SimulatedClock clock(t);

  clock.WakeupWithDeadline(t + absl::Seconds(20),
                           std::make_shared<TestWaiter>(1, &output));
  clock.WakeupWithDeadline(
      t + absl::Seconds(20), std::make_shared<CallbackWaiter>([&]() {
        output.push_back(2);
        clock.WakeupWithDeadline(t + absl::Seconds(15),
                                 std::make_shared<TestWaiter>(3, &output));
      }));
  clock.AdvanceTime(absl::Seconds(20));
  // Both waiters are triggered because the #3 one is already expired when
  // inserted recursively by waiter #2.
  EXPECT_THAT(output, ElementsAre(1, 2, 3));
}

// Verifies that a long taking Wakeup notification results in triggering
// other waiters that expire later.
TEST(SimulatedClockTest, LongRunningWakeup) {
  std::vector<int> output;
  absl::Time t = GetTestInitialTime();
  SimulatedClock clock(t);

  clock.WakeupWithDeadline(t + absl::Seconds(10),
                           std::make_shared<TestWaiter>(1, &output));
  clock.WakeupWithDeadline(
      t + absl::Seconds(20), std::make_shared<CallbackWaiter>([&]() {
        output.push_back(2);
        clock.AdvanceTime(absl::Seconds(10));
      }));
  clock.WakeupWithDeadline(t + absl::Seconds(30),
                           std::make_shared<TestWaiter>(3, &output));
  // Advance time by 20 second, which will advance time by another 10 seconds
  // when waking up waiter #2.
  clock.AdvanceTime(absl::Seconds(20));
  EXPECT_THAT(output, ElementsAre(1, 2, 3));
}

}  // namespace
}  // namespace tensorflow_federated

/*
 * Copyright 2018 Google LLC
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

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_BASE_SCHEDULER_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_BASE_SCHEDULER_H_

/**
 * Overview
 * ========
 *
 * A simple implementation of a scheduler (thread pool). Allows to schedule
 * tasks and futures.
 */

#include <functional>
#include <memory>

#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/move_to_lambda.h"

namespace tensorflow_federated {

/**
 * A Worker allows to schedule tasks which are executed sequentially.
 * Workers are created from a Scheduler.
 *
 * Lifetime and destruction:
 *
 *   - The scheduler from which a worker is created must not be destructed
 *     before the worker.
 *
 *   - The worker must not be destructed before all its tasks are finished.
 */
class Worker {
 public:
  virtual ~Worker() = default;
  Worker() = default;

  Worker(Worker const&) = delete;
  Worker& operator=(Worker const&) = delete;

  /**
   * Schedules a task on this worker. Tasks are executed strictly sequentially
   * in the order they are scheduled.
   */
  virtual void Schedule(std::function<void()> task) = 0;
};

/**
 * A Scheduler which allows to schedule 'tasks'.
 *
 * Lifetime and destruction:
 *
 *   - A Scheduler *must* be idle (no active or pending work) at destruction
 *     time. See WaitUntilIdle.
 *
 *   - Implies: A Scheduler *must not* be destructed by one of its own tasks
 *
 *   - Implies: Task closures may safely hold raw pointers to their thread pool.
 *     They should *not* have ownership (via a smart-pointer or similar).
 */
class Scheduler {
 public:
  virtual ~Scheduler() = default;
  Scheduler() = default;

  Scheduler(Scheduler const&) = delete;
  Scheduler& operator=(Scheduler const&) = delete;

  /**
   * Creates a new Worker based on this scheduler.
   */
  virtual std::unique_ptr<Worker> CreateWorker();

  /**
   * Schedules a task that will execute on the scheduler.
   */
  virtual void Schedule(std::function<void()> task) = 0;

  /**
   * Waits until there are no tasks running or pending.
   *
   * In this state, the thread pool will not restart working until some
   * external entity is scheduling new tasks, as work caused by tasks spawning
   * other tasks has ceased.
   */
  virtual void WaitUntilIdle() = 0;
};

/**
 * Creates a scheduler using a fixed-size pool of threads to run tasks.
 */
std::unique_ptr<Scheduler> CreateThreadPoolScheduler(std::size_t thread_count);

}  // namespace tensorflow_federated

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_AGGREGATION_BASE_SCHEDULER_H_

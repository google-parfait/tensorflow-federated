/* Copyright 2026, The TensorFlow Federated Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
==============================================================================*/

#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_TASK_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_TASK_H_

#include <coroutine>  // NOLINT(build/c++20)
#include <memory>
#include <optional>
#include <thread>  // NOLINT
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"

namespace tensorflow_federated {

namespace internal {

// Resumes a coroutine handle using a thread-local queue to prevent unbounded
// physical stack growth on deep synchronous completion chains.
inline void ResumeWithTrampoline(std::coroutine_handle<> handle) {
  thread_local int depth = 0;
  thread_local std::vector<std::coroutine_handle<>> queue;

  if (depth > 0) {
    queue.push_back(handle);
    return;
  }

  depth++;
  handle.resume();
  while (!queue.empty()) {
    std::coroutine_handle<> next = queue.back();
    queue.pop_back();
    next.resume();
  }
  depth--;
}

template <typename T>
struct SharedState {
  absl::Mutex mutex;
  bool is_ready ABSL_GUARDED_BY(mutex) = false;
  std::optional<T> value ABSL_GUARDED_BY(mutex);
  std::vector<std::coroutine_handle<>> waiters ABSL_GUARDED_BY(mutex);
  absl::Notification notification;

  void SetValue(T val) {
    std::vector<std::coroutine_handle<>> to_resume;
    {
      absl::MutexLock lock(mutex);
      if (is_ready) {
        return;
      }
      value.emplace(std::move(val));
      is_ready = true;
      to_resume = std::move(waiters);
    }
    notification.Notify();
    for (std::coroutine_handle<> h : to_resume) {
      ResumeWithTrampoline(h);
    }
  }
};

template <>
struct SharedState<void> {
  absl::Mutex mutex;
  bool is_ready ABSL_GUARDED_BY(mutex) = false;
  std::vector<std::coroutine_handle<>> waiters ABSL_GUARDED_BY(mutex);
  absl::Notification notification;

  void SetValue() {
    std::vector<std::coroutine_handle<>> to_resume;
    {
      absl::MutexLock lock(mutex);
      if (is_ready) {
        return;
      }
      is_ready = true;
      to_resume = std::move(waiters);
    }
    notification.Notify();
    for (std::coroutine_handle<> h : to_resume) {
      ResumeWithTrampoline(h);
    }
  }
};

// Internal task type used for synchronous waiting in SyncWait(Task<T>&&).
struct SyncDriverTask {
  struct promise_type {
    SyncDriverTask get_return_object() noexcept {
      return SyncDriverTask{
          std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() noexcept {
      LOG(FATAL) << "Unhandled exception in coroutine";
    }
  };

  std::coroutine_handle<promise_type> handle;
};

// Internal task type used for detached background execution in ScheduleTask and
// MakeShared. Destruction of the coroutine frame is automatic upon completion
// via suspend_never.
struct DetachedDriverTask {
  struct promise_type {
    DetachedDriverTask get_return_object() noexcept {
      return DetachedDriverTask{
          std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() noexcept {
      LOG(FATAL) << "Unhandled exception in coroutine";
    }
  };

  std::coroutine_handle<promise_type> handle;
};

}  // namespace internal

// A move-only, single-consumer C++20 coroutine representation with symmetric
// transfer. Execution is lazy: the coroutine suspends at initial_suspend and
// executes when awaited or passed to SyncWait.
template <typename T = void>
class Task;

template <typename T>
class Task {
 public:
  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;

  struct promise_type {
    std::optional<T> value;
    std::coroutine_handle<> continuation;

    Task<T> get_return_object() noexcept {
      return Task<T>(handle_type::from_promise(*this));
    }

    std::suspend_always initial_suspend() noexcept { return {}; }

    struct FinalAwaiter {
      bool await_ready() const noexcept { return false; }
      std::coroutine_handle<> await_suspend(handle_type h) noexcept {
        if (h.promise().continuation) {
          return h.promise().continuation;
        }
        return std::noop_coroutine();
      }
      void await_resume() const noexcept {}
    };

    FinalAwaiter final_suspend() noexcept { return {}; }

    template <typename U>
      requires std::is_convertible_v<U, T>
    void return_value(U&& val) {
      value.emplace(std::forward<U>(val));
    }

    void unhandled_exception() noexcept {
      LOG(FATAL) << "Unhandled exception in coroutine";
    }
  };

  Task() noexcept : handle_(nullptr) {}
  explicit Task(handle_type handle) noexcept : handle_(handle) {}

  ~Task() {
    if (handle_) {
      handle_.destroy();
    }
  }

  Task(const Task&) = delete;
  Task& operator=(const Task&) = delete;

  Task(Task&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }

  Task& operator=(Task&& other) noexcept {
    if (this != &other) {
      if (handle_) {
        handle_.destroy();
      }
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  bool is_ready() const noexcept { return !handle_ || handle_.done(); }

  auto operator co_await() && noexcept {
    struct Awaiter {
      handle_type handle;

      bool await_ready() const noexcept { return !handle || handle.done(); }

      std::coroutine_handle<> await_suspend(
          std::coroutine_handle<> awaiting) noexcept {
        handle.promise().continuation = awaiting;
        return handle;
      }

      T await_resume() { return std::move(*handle.promise().value); }
    };
    return Awaiter{handle_};
  }

 private:
  template <typename U>
  friend U SyncWait(Task<U>&& task);
  friend void SyncWait(Task<void>&& task);

  handle_type handle_;
};

template <>
class Task<void> {
 public:
  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;

  struct promise_type {
    std::coroutine_handle<> continuation;

    Task<void> get_return_object() noexcept {
      return Task<void>(handle_type::from_promise(*this));
    }

    std::suspend_always initial_suspend() noexcept { return {}; }

    struct FinalAwaiter {
      bool await_ready() const noexcept { return false; }
      std::coroutine_handle<> await_suspend(handle_type h) noexcept {
        if (h.promise().continuation) {
          return h.promise().continuation;
        }
        return std::noop_coroutine();
      }
      void await_resume() const noexcept {}
    };

    FinalAwaiter final_suspend() noexcept { return {}; }

    void return_void() noexcept {}

    void unhandled_exception() noexcept {
      LOG(FATAL) << "Unhandled exception in coroutine";
    }
  };

  Task() noexcept : handle_(nullptr) {}
  explicit Task(handle_type handle) noexcept : handle_(handle) {}

  ~Task() {
    if (handle_) {
      handle_.destroy();
    }
  }

  Task(const Task&) = delete;
  Task& operator=(const Task&) = delete;

  Task(Task&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }

  Task& operator=(Task&& other) noexcept {
    if (this != &other) {
      if (handle_) {
        handle_.destroy();
      }
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  bool is_ready() const noexcept { return !handle_ || handle_.done(); }

  auto operator co_await() && noexcept {
    struct Awaiter {
      handle_type handle;

      bool await_ready() const noexcept { return !handle || handle.done(); }

      std::coroutine_handle<> await_suspend(
          std::coroutine_handle<> awaiting) noexcept {
        handle.promise().continuation = awaiting;
        return handle;
      }

      void await_resume() {}
    };
    return Awaiter{handle_};
  }

 private:
  friend void SyncWait(Task<void>&& task);

  handle_type handle_;
};

// A thread-safe, multi-consumer memoizing coroutine handle. Multiple consumers
// can co_await or SyncWait on the same SharedTask concurrently. When ready,
// awaiting yields a `const T&` (or `void`), enabling zero-copy inspection of
// move-only values.
template <typename T = void>
class SharedTask;

template <typename T>
class SharedTask {
 public:
  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;

  struct promise_type {
    std::shared_ptr<internal::SharedState<T>> state =
        std::make_shared<internal::SharedState<T>>();

    SharedTask<T> get_return_object() noexcept { return SharedTask<T>(state); }

    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }

    template <typename U>
      requires std::is_convertible_v<U, T>
    void return_value(U&& val) {
      state->SetValue(std::forward<U>(val));
    }

    void unhandled_exception() noexcept {
      LOG(FATAL) << "Unhandled exception in coroutine";
    }
  };

  SharedTask() = default;
  explicit SharedTask(std::shared_ptr<internal::SharedState<T>> state)
      : state_(std::move(state)) {}

  SharedTask(const SharedTask&) = default;
  SharedTask& operator=(const SharedTask&) = default;
  SharedTask(SharedTask&&) = default;
  SharedTask& operator=(SharedTask&&) = default;

  static SharedTask<T> FromValue(T value) {
    auto state = std::make_shared<internal::SharedState<T>>();
    state->SetValue(std::move(value));
    return SharedTask<T>(std::move(state));
  }

  bool is_ready() const {
    if (state_ == nullptr) {
      return false;
    }
    absl::MutexLock lock(state_->mutex);
    return state_->is_ready;
  }

  auto operator co_await() const noexcept {
    struct Awaiter {
      std::shared_ptr<internal::SharedState<T>> state;

      bool await_ready() const noexcept {
        if (state == nullptr) {
          return true;
        }
        absl::MutexLock lock(state->mutex);
        return state->is_ready;
      }

      bool await_suspend(std::coroutine_handle<> awaiting) noexcept {
        absl::MutexLock lock(state->mutex);
        if (state->is_ready) {
          return false;
        }
        state->waiters.push_back(awaiting);
        return true;
      }

      const T& await_resume() const {
        CHECK(state != nullptr) << "Awaiting null SharedTask";
        absl::MutexLock lock(state->mutex);
        return *state->value;
      }
    };
    return Awaiter{state_};
  }

 private:
  template <typename U>
  friend const U& SyncWait(const SharedTask<U>& task);
  friend void SyncWait(const SharedTask<void>& task);

  template <typename U>
  friend SharedTask<U> ScheduleTask(ThreadPool* pool, Task<U> task);
  template <typename U>
  friend SharedTask<U> MakeShared(Task<U> task);

  std::shared_ptr<internal::SharedState<T>> state_;
};

template <>
class SharedTask<void> {
 public:
  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;

  struct promise_type {
    std::shared_ptr<internal::SharedState<void>> state =
        std::make_shared<internal::SharedState<void>>();

    SharedTask<void> get_return_object() noexcept {
      return SharedTask<void>(state);
    }

    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }

    void return_void() noexcept { state->SetValue(); }

    void unhandled_exception() noexcept {
      LOG(FATAL) << "Unhandled exception in coroutine";
    }
  };

  SharedTask() = default;
  explicit SharedTask(std::shared_ptr<internal::SharedState<void>> state)
      : state_(std::move(state)) {}

  SharedTask(const SharedTask&) = default;
  SharedTask& operator=(const SharedTask&) = default;
  SharedTask(SharedTask&&) = default;
  SharedTask& operator=(SharedTask&&) = default;

  static SharedTask<void> MakeReady() {
    auto state = std::make_shared<internal::SharedState<void>>();
    state->SetValue();
    return SharedTask<void>(std::move(state));
  }

  bool is_ready() const {
    if (state_ == nullptr) {
      return false;
    }
    absl::MutexLock lock(state_->mutex);
    return state_->is_ready;
  }

  auto operator co_await() const noexcept {
    struct Awaiter {
      std::shared_ptr<internal::SharedState<void>> state;

      bool await_ready() const noexcept {
        if (state == nullptr) {
          return true;
        }
        absl::MutexLock lock(state->mutex);
        return state->is_ready;
      }

      bool await_suspend(std::coroutine_handle<> awaiting) noexcept {
        absl::MutexLock lock(state->mutex);
        if (state->is_ready) {
          return false;
        }
        state->waiters.push_back(awaiting);
        return true;
      }

      void await_resume() const {
        CHECK(state != nullptr) << "Awaiting null SharedTask";
      }
    };
    return Awaiter{state_};
  }

 private:
  friend void SyncWait(const SharedTask<void>& task);

  template <typename U>
  friend SharedTask<U> ScheduleTask(ThreadPool* pool, Task<U> task);
  template <typename U>
  friend SharedTask<U> MakeShared(Task<U> task);

  std::shared_ptr<internal::SharedState<void>> state_;
};

// Synchronously waits for a single-consumer `Task<T>` to finish and returns the
// result by moving out of the task.
template <typename T>
T SyncWait(Task<T>&& task) {
  CHECK(task.handle_ != nullptr) << "Cannot SyncWait on empty Task";

  std::optional<T> result;
  absl::Notification done;

  auto sync_driver = [](Task<T> t, std::optional<T>& res,
                        absl::Notification& n) -> internal::SyncDriverTask {
    res.emplace(co_await std::move(t));
    n.Notify();
    co_return;
  };

  internal::SyncDriverTask waiter = sync_driver(std::move(task), result, done);
  waiter.handle.resume();

  done.WaitForNotification();
  waiter.handle.destroy();
  return std::move(*result);
}

inline void SyncWait(Task<void>&& task) {
  CHECK(task.handle_ != nullptr) << "Cannot SyncWait on empty Task";

  absl::Notification done;

  auto sync_driver = [](Task<void> t,
                        absl::Notification& n) -> internal::SyncDriverTask {
    co_await std::move(t);
    n.Notify();
    co_return;
  };

  internal::SyncDriverTask waiter = sync_driver(std::move(task), done);
  waiter.handle.resume();

  done.WaitForNotification();
  waiter.handle.destroy();
}

// Synchronously waits for a `SharedTask<T>` to finish and returns a const
// reference to the memoized result.
template <typename T>
const T& SyncWait(const SharedTask<T>& task) {
  CHECK(task.state_ != nullptr) << "Cannot SyncWait on empty SharedTask";
  task.state_->notification.WaitForNotification();
  absl::MutexLock lock(&task.state_->mutex);
  return *task.state_->value;
}

inline void SyncWait(const SharedTask<void>& task) {
  CHECK(task.state_ != nullptr) << "Cannot SyncWait on empty SharedTask";
  task.state_->notification.WaitForNotification();
}

// Converts a lazy `Task<T>` into an active `SharedTask<T>` executing inline on
// the current thread up to its first suspension point.
template <typename T>
SharedTask<T> MakeShared(Task<T> task) {
  auto state = std::make_shared<internal::SharedState<T>>();

  auto driver = [](Task<T> t, std::shared_ptr<internal::SharedState<T>> s)
      -> internal::DetachedDriverTask {
    std::shared_ptr<internal::SharedState<T>> local_state = std::move(s);
    if constexpr (std::is_void_v<T>) {
      co_await std::move(t);
      std::shared_ptr<internal::SharedState<T>> target = std::move(local_state);
      target->SetValue();
    } else {
      T val = co_await std::move(t);
      std::shared_ptr<internal::SharedState<T>> target = std::move(local_state);
      target->SetValue(std::move(val));
    }
  };

  internal::DetachedDriverTask driver_task = driver(std::move(task), state);
  driver_task.handle.resume();
  return SharedTask<T>(state);
}

// Creates an immediately-ready `SharedTask<T>` with zero coroutine suspension,
// zero thread dispatch, and zero heap-allocated coroutine frames.
template <typename T>
SharedTask<std::decay_t<T>> MakeReadySharedTask(T&& val) {
  using CleanT = std::decay_t<T>;
  auto state = std::make_shared<internal::SharedState<CleanT>>();
  state->SetValue(std::forward<T>(val));
  return SharedTask<CleanT>(std::move(state));
}

inline SharedTask<void> MakeReadySharedTask() {
  auto state = std::make_shared<internal::SharedState<void>>();
  state->SetValue();
  return SharedTask<void>(std::move(state));
}

// Schedules a lazy `Task<T>` onto `pool` (or a detached thread if pool is null)
// and returns a `SharedTask<T>`. When the task suspends for dependencies, the
// worker thread is not blocked, eliminating thread starvation and deadlock.
template <typename T>
SharedTask<T> ScheduleTask(ThreadPool* pool, Task<T> task) {
  auto state = std::make_shared<internal::SharedState<T>>();

  auto driver = [](Task<T> t, std::shared_ptr<internal::SharedState<T>> s)
      -> internal::DetachedDriverTask {
    std::shared_ptr<internal::SharedState<T>> local_state = std::move(s);
    if constexpr (std::is_void_v<T>) {
      co_await std::move(t);
      std::shared_ptr<internal::SharedState<T>> target = std::move(local_state);
      target->SetValue();
    } else {
      T val = co_await std::move(t);
      std::shared_ptr<internal::SharedState<T>> target = std::move(local_state);
      target->SetValue(std::move(val));
    }
  };

  internal::DetachedDriverTask driver_task = driver(std::move(task), state);
  std::coroutine_handle<> handle = driver_task.handle;

  auto run_task = [handle]() mutable { handle.resume(); };

  if (pool != nullptr) {
    absl::Status status = pool->Schedule(std::move(run_task));
    if (!status.ok()) {
      handle.destroy();
      if constexpr (std::is_constructible_v<T, absl::Status>) {
        state->SetValue(T(status));
      } else {
        LOG(FATAL) << "Failed to schedule task on ThreadPool: " << status;
      }
    }
  } else {
    std::thread th(std::move(run_task));
    th.detach();
  }

  return SharedTask<T>(state);
}

namespace internal {

inline absl::Status MakeStatus(
    absl::Status status
) {
  return status;
}

}  // namespace internal

}  // namespace tensorflow_federated

// Coroutine-safe error-handling macros.
#define TFF_CO_RETURN_IF_ERROR(expr)                                           \
  do {                                                                         \
    const absl::Status& __tff_co_status = (expr);                              \
    if (!__tff_co_status.ok()) {                                               \
      co_return ::tensorflow_federated::internal::MakeStatus(__tff_co_status); \
    }                                                                          \
  } while (false)

#define TFF_CO_ASSIGN_OR_RETURN_IMPL_(var, decl, expr)      \
  auto var = (expr);                                        \
  if (!var.ok()) {                                          \
    co_return ::tensorflow_federated::internal::MakeStatus( \
        std::move(var).status());                           \
  }                                                         \
  decl = *std::move(var)

#define TFF_CO_ASSIGN_OR_RETURN(decl, expr) \
  TFF_CO_ASSIGN_OR_RETURN_IMPL_(__tff_co_res_##__LINE__, decl, expr)

#define TFF_CO_BIND_OR_RETURN_IMPL_(var, decl, expr)                      \
  const auto& var = (expr);                                               \
  if (!var.ok()) {                                                        \
    co_return ::tensorflow_federated::internal::MakeStatus(var.status()); \
  }                                                                       \
  decl = *var

#define TFF_CO_BIND_OR_RETURN(decl, expr) \
  TFF_CO_BIND_OR_RETURN_IMPL_(__tff_co_ref_##__LINE__, decl, expr)

#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_TASK_H_

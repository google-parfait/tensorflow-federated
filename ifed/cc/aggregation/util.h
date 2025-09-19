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

#pragma once

#include <jni.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/fixed_array.h"
#include "absl/log/absl_log.h"
#include "absl/base/attributes.h"

// Checks whether a JNI call failed. This is meant to be called right after any
// JNI call, to detect error conditions as early as possible. For functions that
// need to return "void" on failure you can specify "void()" as the return_val.
#define JNI_FAILURE_CHECK(env, return_val)              \
  while (ABSL_PREDICT_FALSE((env)->ExceptionCheck())) { \
    env->ExceptionClear();                              \
    return return_val;                                  \
  }

/**
 * Check that the condition holds, otherwise die. Any additional messages can
 * be streamed into the invocation. Example:
 *
 *     JNI_CHECK(condition) << "stuff went wrong";
 */
#define JNI_CHECK(condition)                         \
  ABSL_LOG_IF(FATAL, ABSL_PREDICT_FALSE(!(condition))) \
      << ("Check failed: " #condition ". ")

namespace jni {

static absl::Status CheckJniException(JNIEnv* env, absl::string_view context) {
  if (env->ExceptionCheck()) {
    env->ExceptionDescribe();  // Optional: prints Java stack trace to stderr
    env->ExceptionClear();
    return absl::InternalError("Java JNI exception:" + std::string(context));
  }

  return absl::OkStatus();
}

static inline absl::StatusOr<std::string> JbyteArrayToString(JNIEnv* env, jbyteArray arr) {
  int len = env->GetArrayLength(arr);
  char* buf = new char[len];
  std::unique_ptr<char[]> buf_uptr(buf);
  env->GetByteArrayRegion(arr, 0, len, reinterpret_cast<jbyte*>(buf));
  if (auto status = CheckJniException(env, "GetByteArrayRegion"); !status.ok()) {
    return status;
  }

  return std::string(buf, len);
}

// Throws an exception of the given class. The exception is expected to have
// a two-argument constructor, where the first argument represents a canonical
// error code, and the second argument is the exception message.
//
// If errors occur during exception construction, a runtime exception
// will be thrown instead, or if that fails, the process will be aborted.
// After this method returns an exception will always be set in the JNI env.
static void ThrowCustomStatusCodeException(JNIEnv* env,
                                           const std::string& exception_class,
                                           int code,
                                           const std::string& message) {
  jclass excl = env->FindClass(exception_class.c_str());
  JNI_FAILURE_CHECK(env, void());
  jmethodID ctor = env->GetMethodID(excl, "<init>", "(ILjava/lang/String;)V");
  JNI_FAILURE_CHECK(env, void());
  jstring message_object = env->NewStringUTF(message.c_str());
  JNI_FAILURE_CHECK(env, void());
  jthrowable ex =
          (jthrowable)(env->NewObject(excl, ctor, code, message_object));
  JNI_FAILURE_CHECK(env, void());
  env->Throw(ex);
}

// Creates a JNIEnv via the passed JavaVM*, attaching the current thread if it
// is not already. If an attach was needed, detaches when this is destroyed.
//
// ScopedJniEnv must not be shared among threads and destructs on the same
// thread.
class ScopedJniEnv final {
 public:
  explicit ScopedJniEnv(JavaVM* jvm)
      : jvm_(jvm), env_(nullptr), is_attached_(false) {
    // We don't make any assumptions about the state of the current thread, and
    // we want to leave it in the state we received it with respect to the
    // JavaVm. So we only attach and detach when needed, and we always delete
    // local references.
    jint error = jvm_->GetEnv(reinterpret_cast<void**>(&env_), JNI_VERSION_1_2);
    if (error != JNI_OK) {
      error = AttachCurrentThread(jvm_, &env_);
      JNI_CHECK(error == JNI_OK);
      is_attached_ = true;
    }
  }

  virtual ~ScopedJniEnv() {
    if (is_attached_) {
      (void)jvm_->DetachCurrentThread();
    }
  }

  JNIEnv* env() { return env_; }

 private:
  template <typename JNIEnvArgType>
  static jint AttachCurrentThreadImpl(JavaVM* vm,
                                      jint (JavaVM::*fn)(JNIEnvArgType, void*),
                                      JNIEnv** env) {
    static_assert(std::is_same_v<JNIEnvArgType, void**> ||
                  std::is_same_v<JNIEnvArgType, JNIEnv**>);
    return (vm->*fn)(reinterpret_cast<JNIEnvArgType>(env), nullptr);
  }

  static jint AttachCurrentThread(JavaVM* vm, JNIEnv** env) {
    // The NDK and JDK versions of jni.h disagree on the signatures for the
    // JavaVM::AttachCurrentThread member function (the former uses 'JavaVM*'
    // and the latter uses 'void**'). To avoid causing linker errors when the
    // JDK's jni.h is accidentally put on the include path during an Android
    // build, we use the indirection below when calling the function. It's not
    // sufficient to #ifdef around __ANDROID__, because whatever is including
    // this header file might put the JDK jni.h version on the include path.
    return AttachCurrentThreadImpl(vm, &JavaVM::AttachCurrentThread, env);
  }

  ScopedJniEnv(const ScopedJniEnv&) = delete;
  void operator=(const ScopedJniEnv&) = delete;

  JavaVM* jvm_;
  JNIEnv* env_;
  bool is_attached_;
};

// Parses a proto from a Java byte array.
//
// If any JNI calls fail, or if the parsing of the proto fails, then this
// JNI_CHECK-fails.
//
// This method does not call `JNIEnv::DeleteLocalRef` on the given `jbyteArray`.
//
// This is meant to be used as a convenient way to use serialized protobufs as
// part of a JNI API contract, since in such cases we can safely assume that the
// input argument will always be a valid proto (and anything else would be a
// programmer error).
template <typename MessageT>
static absl::StatusOr<MessageT> ParseProtoFromJByteArray(JNIEnv* env, jbyteArray byte_array) {
  jsize length = env->GetArrayLength(byte_array);
  if (auto status = CheckJniException(env, "GetArrayLength"); !status.ok()) {
    return status;
  }

  if (length >= INT32_MAX) {
    return absl::InvalidArgumentError("Byte array too large");
  }

  if (length < 0) {
    return absl::InvalidArgumentError("Negative length");
  }

  if (length == 0) {
    return MessageT();
  }

  MessageT result;
  // This will make a copy of the data into buffer, but generally the proto data
  // will small enough that this shouldn't matter.
  absl::FixedArray<jbyte> buffer(length);
  env->GetByteArrayRegion(byte_array, 0, length, buffer.data());
  if (auto status = CheckJniException(env, "GetByteArrayRegion"); !status.ok()) {
    return status;
  }

  if (!result.ParseFromArray(buffer.data(), length)) {
    return absl::InvalidArgumentError("Failed to parse proto");
  }

  return result;
}

// Serializes a proto to a `jbyteArray`.
//
// The caller must call `JNIEnv::DeleteLocalRef` on the returned `jbyteArray`
// once it is done with it.
//
// If any JNI calls fail, then this JNI_CHECK-fails.
template <typename MessageT>
static absl::StatusOr<jbyteArray> SerializeProtoToJByteArray(
  JNIEnv* env,
  const MessageT& proto
) {
  int length = static_cast<int>(proto.ByteSizeLong());
  jbyteArray byte_array = env->NewByteArray(length);
  if (auto status = CheckJniException(env, "NewByteArray"); !status.ok()) {
    return status;
  }

  if (byte_array == nullptr) {
    return absl::InternalError("Failed to allocate byte array");
  }

  // This serializes into a buffer and then copies that buffer to the Java byte
  // array. The proto data is generally small enough that this extra copy
  // shouldn't matter.
  absl::FixedArray<jbyte> buffer(length);
  if (!proto.SerializeToArray(buffer.data(), length)) {
    return absl::InternalError("Failed to serialize proto");
  }

  env->SetByteArrayRegion(byte_array, 0, length, buffer.data());
  if (auto status = CheckJniException(env, "SetByteArrayRegion"); !status.ok()) {
    return status;
  }

  return byte_array;
}

// Describes the method name and JNI method signature of a Java callback.
struct JavaMethodSig {
  char const* name;
  char const* signature;
};
// Describes the field name and JNI type signature of a Java field.
struct JavaFieldSig {
  char const* name;
  char const* signature;
};

// A utility for ensuring that a local JNI reference is deleted once the object
// goes out of scope. This class is only intended to be used inside a function
// body (and not to be returned or passed as an argument).
class LocalRefDeleter {
 public:
  LocalRefDeleter(JNIEnv* env, jobject local_ref)
      : env_(env), local_ref_(local_ref) {}
  // Prevent copies & moves, to make it harder to accidentally have this object
  // be passed as a parameter or return type.
  LocalRefDeleter(LocalRefDeleter& other) = delete;
  LocalRefDeleter(LocalRefDeleter&& other) = delete;
  ~LocalRefDeleter() { env_->DeleteLocalRef(local_ref_); }

 private:
  JNIEnv* env_;
  jobject local_ref_;
};

}  // namespace jni

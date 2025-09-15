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

#include <jni.h>
#include <string.h>

#include "absl/status/statusor.h"
#include "util.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/tensorflow_checkpoint_builder_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/tensorflow_checkpoint_parser_factory.h"

using tff::jni::JbyteArrayToString;
using tff::jni::ParseProtoFromJByteArray;
using tff::jni::ThrowCustomStatusCodeException;

#define JFUN(METHOD_NAME) \
  Java_com_google_fcp_aggregation_AggregationSession_##METHOD_NAME

#define AG_EXCEPTION_CLASS "com/google/fcp/aggregation/AggregationException"

// Helper methods
// ==============
namespace {

// Throws a AggregationException with the given status code and message in the
// JNI environment.
void ThrowAggregationException(JNIEnv* env, int code,
                               const std::string& message) {
  ThrowCustomStatusCodeException(env, AG_EXCEPTION_CLASS, code, message);
}
}  // namespace

// JNI bindings
// ============

extern "C" JNIEXPORT jlong JNICALL JFUN(createNativeFromByteArray)(
    JNIEnv* env, jclass, jbyteArray configurationByteArray) {
  absl::StatusOr<tensorflow_federated::aggregation::Configuration> config =
      ParseProtoFromJByteArray<
          tensorflow_federated::aggregation::Configuration>(
          env, configurationByteArray);
  if (!config.ok()) {
    ThrowAggregationException(env, config.status().raw_code(),
                              std::string(config.status().message()));
    return 0;
  }

  absl::StatusOr<
      std::unique_ptr<tensorflow_federated::aggregation::CheckpointAggregator>>
      result = tensorflow_federated::aggregation::CheckpointAggregator::Create(
          config.value());
  if (!result.ok()) {
    ThrowAggregationException(env, result.status().raw_code(),
                              std::string(result.status().message()));
    return 0;
  }

  return reinterpret_cast<jlong>(result.value().release());
}

extern "C" JNIEXPORT void JNICALL JFUN(mergeWith)(
    JNIEnv* env, jclass, jlong handle, jbyteArray configurationByteArray,
    jobjectArray serializedStates) {
  if (handle == 0) {
    ThrowAggregationException(env, /* code= */ 2,
                              "Invalid session handle (session closed?)");
    return;
  }
  tensorflow_federated::aggregation::CheckpointAggregator* aggregator(
      reinterpret_cast<
          tensorflow_federated::aggregation::CheckpointAggregator*>(handle));

  absl::StatusOr<tensorflow_federated::aggregation::Configuration> config =
      ParseProtoFromJByteArray<
          tensorflow_federated::aggregation::Configuration>(
          env, configurationByteArray);
  if (!config.ok()) {
    ThrowAggregationException(env, config.status().raw_code(),
                              std::string(config.status().message()));
    return;
  }

  int len = env->GetArrayLength(serializedStates);
  for (int i = 0; i < len; i++) {
    jbyteArray serializedState =
        (jbyteArray)env->GetObjectArrayElement(serializedStates, i);
    absl::StatusOr<std::unique_ptr<
        tensorflow_federated::aggregation::CheckpointAggregator>>
        result = tensorflow_federated::aggregation::CheckpointAggregator::
            Deserialize(config.value(),
                        JbyteArrayToString(env, serializedState));
    if (!result.ok()) {
      ThrowAggregationException(env, result.status().raw_code(),
                                std::string(result.status().message()));
      return;
    }

    absl::Status status = aggregator->MergeWith(std::move(*(result.value())));
    if (!status.ok()) {
      ThrowAggregationException(env, status.raw_code(),
                                std::string(status.message()));
      return;
    }
  }
  return;
}

// Note: This method consumes (and destroys) the Aggregation session.  After
// this method is called, the caller should never re-use the Aggregation
// session.
extern "C" JNIEXPORT void JNICALL JFUN(closeNative)(JNIEnv* env, jobject obj,
                                                    jlong handle) {
  if (handle == 0) {
    ThrowAggregationException(env, /* code= */ 2,
                              "Invalid session handle (session closed?)");
    return;
  }
  tensorflow_federated::aggregation::CheckpointAggregator* aggregator(
      reinterpret_cast<
          tensorflow_federated::aggregation::CheckpointAggregator*>(handle));
  delete aggregator;
}

extern "C" JNIEXPORT void JNICALL JFUN(runAccumulate)(
    JNIEnv* env, jclass, jlong handle, jobjectArray checkpoints) {
  if (handle == 0) {
    ThrowAggregationException(env, /* code= */ 2,
                              "Invalid session handle (session closed?)");
    return;
  }

  int len = env->GetArrayLength(checkpoints);
  for (int i = 0; i < len; i++) {
    jbyteArray checkpoint =
        (jbyteArray)env->GetObjectArrayElement(checkpoints, i);

    absl::Cord cord(JbyteArrayToString(env, checkpoint));
    tensorflow_federated::aggregation::CheckpointAggregator* aggregator(
        reinterpret_cast<
            tensorflow_federated::aggregation::CheckpointAggregator*>(handle));
    tensorflow_federated::aggregation::tensorflow::
        TensorflowCheckpointParserFactory parser_factory;
    absl::StatusOr<
        std::unique_ptr<tensorflow_federated::aggregation::CheckpointParser>>
        parser = parser_factory.Create(cord);
    if (!parser.ok()) {
      ThrowAggregationException(env, parser.status().raw_code(),
                                std::string(parser.status().message()));
      return;
    }
    absl::Status status = aggregator->Accumulate(*(parser.value()));
    if (!status.ok()) {
      ThrowAggregationException(env, status.raw_code(),
                                std::string(status.message()));
      return;
    }
  }
  return;
}

extern "C" JNIEXPORT jbyteArray JNICALL JFUN(runReport)(JNIEnv* env, jclass,
                                                        jlong handle) {
  if (handle == 0) {
    ThrowAggregationException(env, 2,
                              "Invalid session handle (session closed?)");
    return {};
  }
  tensorflow_federated::aggregation::CheckpointAggregator* aggregator(
      reinterpret_cast<
          tensorflow_federated::aggregation::CheckpointAggregator*>(handle));
  tensorflow_federated::aggregation::tensorflow::
      TensorflowCheckpointBuilderFactory builder_factory;
  std::unique_ptr<tensorflow_federated::aggregation::CheckpointBuilder>
      builder = builder_factory.Create();
  absl::Status status = aggregator->Report(*builder);
  if (!status.ok()) {
    ThrowAggregationException(env, status.raw_code(),
                              std::string(status.message()));
    return {};
  }
  absl::StatusOr<absl::Cord> res = builder->Build();
  if (!res.ok()) {
    ThrowAggregationException(env, res.status().raw_code(),
                              std::string(res.status().message()));
    return {};
  }
  std::string result = std::string(res.value());
  absl::CopyCordToString(*res, &result);
  jbyteArray ret = env->NewByteArray(result.length());
  env->SetByteArrayRegion(ret, 0, result.length(),
                          reinterpret_cast<const jbyte*>(result.c_str()));
  return ret;
}

extern "C" JNIEXPORT jbyteArray JNICALL JFUN(serialize)(JNIEnv* env, jclass,
                                                        jlong handle) {
  if (handle == 0) {
    ThrowAggregationException(env, 2,
                              "Invalid session handle (session closed?)");
    return {};
  }
  tensorflow_federated::aggregation::CheckpointAggregator* aggregator(
      reinterpret_cast<
          tensorflow_federated::aggregation::CheckpointAggregator*>(handle));
  absl::StatusOr<std::string> status = std::move(*aggregator).Serialize();
  if (!status.ok()) {
    ThrowAggregationException(env, status.status().raw_code(),
                              std::string(status.status().message()));
    return {};
  }

  std::string result = status.value();
  jbyteArray ret = env->NewByteArray(result.length());
  env->SetByteArrayRegion(ret, 0, result.length(),
                          reinterpret_cast<const jbyte*>(result.c_str()));
  return ret;
}

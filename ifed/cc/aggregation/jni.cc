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
#include "absl/status/status.h"
#include "util.h"
#include "ifed/engine/cc/plan.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/tensorflow_checkpoint_builder_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/tensorflow_checkpoint_parser_factory.h"

#define JFUN(METHOD_NAME) \
  Java_com_google_tff_aggregation_AggregationSession_##METHOD_NAME

constexpr const char* AG_EXCEPTION_CLASS = "com/google/tff/aggregation/AggregationException";

// Helper methods
// ==============
namespace {

using namespace tensorflow_federated::aggregation;
using IntrinsicArg = tensorflow_federated::aggregation::Configuration_IntrinsicConfig_IntrinsicArg;
using IntrinsicConfig = tensorflow_federated::aggregation::Configuration_IntrinsicConfig;
using tff::jni::JbyteArrayToString;
using tff::jni::ParseProtoFromJByteArray;
using tff::jni::ThrowCustomStatusCodeException;
using tff::jni::CheckJniException;
using tff::jni::SerializeProtoToJByteArray;
using tensorflow_federated::aggregation::tensorflow::TensorflowCheckpointParserFactory;
using tensorflow_federated::aggregation::tensorflow::TensorflowCheckpointBuilderFactory;

// Throws a AggregationException with the given status code and message in the
// JNI environment.
void ThrowAggregationException(JNIEnv* env, int code, const std::string& message) {
  ThrowCustomStatusCodeException(env, AG_EXCEPTION_CLASS, code, message);
}

void ThrowAggregationException(JNIEnv* env, const absl::Status& error) {
  ThrowAggregationException(env, (int)error.code(), std::string(error.message()));
}

std::string Message(const absl::Status& status) {
  return std::string(status.message());
}

// @warning: client-side must check handle != 0
absl::StatusOr<CheckpointAggregator*> AsAggregator(jlong handle) {
  if (handle == 0) {
    return absl::InvalidArgumentError("Invalid session handle (session closed?)");
  }

  return reinterpret_cast<CheckpointAggregator*>(handle);
}

absl::StatusOr<DataType> ConvertDataType(::tensorflow::DataType dtype) {
  switch (dtype) {
    case ::tensorflow::DT_INVALID:
      return DT_INVALID;
    case ::tensorflow::DT_FLOAT:
      return DT_FLOAT;
    case ::tensorflow::DT_DOUBLE:
      return DT_DOUBLE;
    case ::tensorflow::DT_INT32:
      return DT_INT32;
    case ::tensorflow::DT_STRING:
      return DT_STRING;
    case ::tensorflow::DT_INT64:
      return DT_INT64;
    case ::tensorflow::DT_UINT64:
      return DT_UINT64;
    default:
      return absl::InvalidArgumentError("Unsupported dtype: " + std::to_string(dtype));
  }
}

absl::StatusOr<TensorSpecProto> ConvertTensorSpec(const ::tensorflow::TensorSpecProto& tensor) {
  const auto dtype = ConvertDataType(tensor.dtype());
  if (!dtype.ok()) {
    return absl::InvalidArgumentError("Failed to convert data type: " + Message(dtype.status()));
  }

  TensorSpecProto result;
  result.set_name(tensor.name());
  result.set_dtype(dtype.value());
  for (const auto& dim : tensor.shape().dim()) {
    result.mutable_shape()->add_dim_sizes(dim.size());
  }

  return result;
}

absl::StatusOr<IntrinsicArg> ConvertIntrinsicArg(const ifed::engine::tff::ServerAggregationConfig_IntrinsicArg& arg) {
  if (arg.has_state_tensor()) {
    return absl::InvalidArgumentError("State tensors are not supported yet.");
  }

  IntrinsicArg result;
  const auto input_tensor = ConvertTensorSpec(arg.input_tensor());
  if (!input_tensor.ok()) {
    return absl::InvalidArgumentError("Failed to convert input tensor spec: " + Message(input_tensor.status()));
  }
  result.mutable_input_tensor()->CopyFrom(input_tensor.value());
  return result;
}

absl::StatusOr<IntrinsicConfig> ConvertConfig(const ifed::engine::tff::ServerAggregationConfig& config) {
  IntrinsicConfig result;
  for (const auto& aggregation : config.inner_aggregations()) {
    const auto converted = ConvertConfig(aggregation);
    if (!converted.ok()) {
      return absl::InvalidArgumentError("Failed to convert inner aggregation config: " + Message(converted.status()));
    }

    *result.add_inner_intrinsics() = converted.value();
  }

  for (const auto& output : config.output_tensors()) {
    const auto converted = ConvertTensorSpec(output);
    if (!converted.ok()) {
      return absl::InvalidArgumentError("Failed to convert output tensor spec: " + Message(converted.status()));
    }

    *result.add_output_tensors() = converted.value();
  }

  for (const auto& intrinsic_arg : config.intrinsic_args()) {
    const auto converted = ConvertIntrinsicArg(intrinsic_arg);
    if (!converted.ok()) {
      return absl::InvalidArgumentError("Failed to convert intrinsic arg: " + Message(converted.status()));
    }

    *result.add_intrinsic_args() = converted.value();
  }

  result.set_intrinsic_uri(config.intrinsic_uri());
  return result;
}

absl::StatusOr<Configuration>
ExtractAggregationConfigurationFromPlan(const ifed::engine::tff::Plan& plan) {
  if (plan.phase_size() == 0) {
    return absl::Status(absl::StatusCode::kInvalidArgument, "No phases in the plan.");
  }

  if (!plan.phase(0).has_server_phase_v2()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, "No server phases in the plan.");
  }

  Configuration result;
  const auto& server_phase_v2 = plan.phase(0).server_phase_v2();
  for (const auto& config : server_phase_v2.aggregations()) {
    const auto converted = ConvertConfig(config);
    if (!converted.ok()) {
      return absl::Status(converted.status().code(), "Failed to convert aggregation config: " + Message(converted.status()));
    }

    *result.add_intrinsic_configs() = converted.value();
  }

  return result;
}

}  // namespace

// JNI bindings
// ============

extern "C" JNIEXPORT jlong JNICALL JFUN(createNativeFromByteArray)(
    JNIEnv* env, jclass, jbyteArray configurationByteArray) {
  absl::StatusOr<Configuration> config =
      ParseProtoFromJByteArray<
          Configuration>(
          env, configurationByteArray);
  if (!config.ok()) {
    ThrowAggregationException(env, config.status());
    return 0;
  }

  absl::StatusOr<
      std::unique_ptr<CheckpointAggregator>>
      result = CheckpointAggregator::Create(
          config.value());
  if (!result.ok()) {
    ThrowAggregationException(env, result.status());
    return 0;
  }

  return reinterpret_cast<jlong>(result.value().release());
}

extern "C" JNIEXPORT void JNICALL JFUN(mergeWith)(
  JNIEnv* env,
  jclass,
  jlong handle,
  jbyteArray configurationByteArray,
  jobjectArray serializedStates
) {
  auto aggregator = AsAggregator(handle);
  if (!aggregator.ok()) {
    ThrowAggregationException(env, aggregator.status());
    return;
  }

  auto config = ParseProtoFromJByteArray<Configuration>(env, configurationByteArray);
  if (!config.ok()) {
    ThrowAggregationException(env, config.status());
    return;
  }

  int len = env->GetArrayLength(serializedStates);
  for (int i = 0; i < len; i++) {
    jbyteArray serializedState = (jbyteArray)env->GetObjectArrayElement(serializedStates, i);
    if (CheckJniException(env, "GetObjectArrayElement") != absl::OkStatus()) {
      ThrowAggregationException(env,  absl::InternalError("Failed to get array element"));
      return;
    }

    auto serializedStateStr = JbyteArrayToString(env, serializedState);
    if (!serializedStateStr.ok()) {
      ThrowAggregationException(env, serializedStateStr.status());
      return;
    }

    auto other_aggregator = CheckpointAggregator::Deserialize(config.value(), serializedStateStr.value());

    if (!other_aggregator.ok()) {
      ThrowAggregationException(env, other_aggregator.status());
      return;
    }

    if (auto status = aggregator.value()->MergeWith(std::move(*(other_aggregator.value()))); !status.ok()) {
      ThrowAggregationException(env, status);
      return;
    }
  }
  return;
}

// Note: This method consumes (and destroys) the Aggregation session.  After
// this method is called, the caller should never re-use the Aggregation
// session.
extern "C" JNIEXPORT void JNICALL JFUN(closeNative)(JNIEnv* env, jobject obj, jlong handle) {
  auto aggregator = AsAggregator(handle);
  if (!aggregator.ok()) {
    ThrowAggregationException(env, aggregator.status());
    return;
  }

  delete aggregator.value();
}

extern "C" JNIEXPORT void JNICALL JFUN(runAccumulate)(
  JNIEnv* env,
  jclass, jlong handle,
  jobjectArray checkpoints
) {
  auto aggregator = AsAggregator(handle);
  if (!aggregator.ok()) {
    ThrowAggregationException(env, aggregator.status());
    return;
  }

  const auto len = env->GetArrayLength(checkpoints);
  if (auto status = CheckJniException(env, "Failed to get array length"); !status.ok()) {
    ThrowAggregationException(env, status);
    return;
  }

  for (int i = 0; i < len; i++) {
    jbyteArray checkpoint = (jbyteArray)env->GetObjectArrayElement(checkpoints, i);
    if (auto status = CheckJniException(env, "GetObjectArrayElement"); !status.ok()) {
      ThrowAggregationException(env, status);
      return;
    }

    auto checkpointBytes = JbyteArrayToString(env, checkpoint);
    if (!checkpointBytes.ok()) {
      ThrowAggregationException(env, checkpointBytes.status());
      return;
    }

    absl::Cord cord(std::move(checkpointBytes.value()));
    TensorflowCheckpointParserFactory parser_factory;
    auto parser = parser_factory.Create(cord);
    if (!parser.ok()) {
      ThrowAggregationException(env, parser.status());
      return;
    }

    if (auto status = aggregator.value()->Accumulate(*(parser.value())); !status.ok()) {
      ThrowAggregationException(env, status);
      return;
    }
  }
  return;
}

extern "C" JNIEXPORT jbyteArray JNICALL JFUN(runReport)(
  JNIEnv* env,
  jclass,
  jlong handle
) {
  auto aggregator = AsAggregator(handle);
  if (!aggregator.ok()) {
    ThrowAggregationException(env, aggregator.status());
    return {};
  }

  TensorflowCheckpointBuilderFactory builder_factory;
  auto builder = builder_factory.Create();
  absl::Status status = aggregator.value()->Report(*builder);
  if (!status.ok()) {
    ThrowAggregationException(env, status);
    return {};
  }

  auto res = builder->Build();
  if (!res.ok()) {
    ThrowAggregationException(env, res.status());
    return {};
  }

  std::string result;
  absl::CopyCordToString(*res, &result);
  jbyteArray ret = env->NewByteArray(result.length());
  if (auto status = CheckJniException(env, "NewByteArray"); !status.ok()) {
    ThrowAggregationException(env, status);
    return {};
  }

  env->SetByteArrayRegion(ret, 0, result.length(), reinterpret_cast<const jbyte*>(result.c_str()));
  if (auto status = CheckJniException(env, "SetByteArrayRegion"); !status.ok()) {
    ThrowAggregationException(env, status);
    return {};
  }

  return ret;
}

extern "C" JNIEXPORT jbyteArray JNICALL JFUN(serialize)(
  JNIEnv* env,
  jclass,
  jlong handle
) {
  auto aggregator = AsAggregator(handle);
  if (!aggregator.ok()) {
    ThrowAggregationException(env, aggregator.status());
    return {};
  }

  auto serialized = std::move(*aggregator.value()).Serialize();
  if (!serialized.ok()) {
    ThrowAggregationException(env, serialized.status());
    return {};
  }

  const auto serializedAggregator = serialized.value();
  auto byteArray = env->NewByteArray(serializedAggregator.length());
  if (auto status = CheckJniException(env, "NewByteArray"); !status.ok()) {
    ThrowAggregationException(env, status);
    return {};
  }

  const auto aggregatorBytes = reinterpret_cast<const jbyte*>(serializedAggregator.c_str());
  env->SetByteArrayRegion(byteArray, 0, serializedAggregator.length(), aggregatorBytes);
  if (auto status = CheckJniException(env, "SetByteArrayRegion"); !status.ok()) {
    ThrowAggregationException(env, status);
    return {};
  }

  return byteArray;
}

extern "C" JNIEXPORT jbyteArray JNICALL JFUN(extractConfiguration)(
  JNIEnv* env,
  jclass,
  jbyteArray planByteArray
) {
  const auto plan = ParseProtoFromJByteArray<ifed::engine::tff::Plan>(env, planByteArray);
  if (!plan.ok()) {
    ThrowAggregationException(env, plan.status());
    return {};
  }

  const auto config = ExtractAggregationConfigurationFromPlan(plan.value());
  if (!config.ok()) {
    ThrowAggregationException(env, config.status());
    return {};
  }

  const auto result = tff::jni::SerializeProtoToJByteArray(env, config.value());
  if (!result.ok()) {
    ThrowAggregationException(env, result.status());
    return {};
  }

  return result.value();
}

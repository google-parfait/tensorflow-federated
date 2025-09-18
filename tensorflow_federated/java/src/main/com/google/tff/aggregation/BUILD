# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
package(
    default_visibility = ["//visibility:public"],
)

java_library(
    name = "aggregation",
    srcs = [
        "AggregationSession.java",
    ],
    runtime_deps = [
        "//tensorflow_federated/cc/core/impl/aggregation/jni:libaggregation-jni_darwin",
    ],
    deps = [
        ":aggregation_exception",
        ":native_handle",
        "@maven//:com_google_code_findbugs_jsr305",
        "@maven//:com_google_guava_guava",

        #"//fcp/aggregation/core:tensor_java_proto",
        #"//fcp/aggregation/protocol:configuration_java_proto",
        # "//fcp/java_src/main/java/com/google/fcp/tensorflow:protos_all_java_proto",
        # "//fcp/protos:plan_java_proto",
        # "@com_google_protobuf//java/core",

    ],
)

java_library(
    name = "aggregation_exception",
    srcs = ["AggregationException.java"],
    deps = [],
)

java_library(
    name = "native_handle",
    srcs = ["NativeHandle.java"],
    deps = ["@maven//:com_google_guava_guava"],
)

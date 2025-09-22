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
    resources = select({
        "@platforms//os:macos": [":copy_aggregation_jni_to_resources_dylib"],
        "@platforms//os:linux": [":copy_aggregation_jni_to_resources_so"],
        "//conditions:default": [],
    }),
		resource_strip_prefix = "engine/java/src/main/org/jetbrains/tff/engine",
    deps = [
        ":aggregation_exception",
        ":native_handle",
        "@maven//:com_google_code_findbugs_jsr305",
        "@maven//:com_google_guava_guava",
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

genrule(
    name = "copy_aggregation_jni_to_resources_dylib",
    srcs = ["//engine/cc/aggregation:aggregation-jni"],
    outs = ["libaggregation-jni.dylib"],
    cmd = "cp $< $@",
    target_compatible_with = ["@platforms//os:macos"],
)

genrule(
    name = "copy_aggregation_jni_to_resources_so",
    srcs = ["//engine/cc/aggregation:aggregation-jni"],
    outs = ["libaggregation-jni.so"],
    cmd = "cp $< $@",
    target_compatible_with = ["@platforms//os:linux"],
)

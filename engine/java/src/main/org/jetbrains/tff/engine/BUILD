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

load("@rules_jvm_external//:defs.bzl", "java_export")
load("@rules_java//java:java_library.bzl", "java_library")

TFF_ENGINE_VERSION = "1.0.0"
TFF_ENGINE_PACKAGE = "org.jetbrains:tff-engine-{os}:{version}"

java_library(
    name = "engine",
    srcs = [
        "AggregationSession.java",
        "NativeLibraryLoader.java",
        "AggregationException.java",
        "NativeHandle.java",
    ],
    resources = [":copy_aggregation_jni_to_resources"],
	resource_strip_prefix = "engine/java/src/main/org/jetbrains/tff/engine",
    deps = [
        "@maven//:com_google_code_findbugs_jsr305",
        "@maven//:com_google_guava_guava",
    ],
)

java_export(
    name = "engine_package_osx",
    exports = [":engine"],
    maven_coordinates = TFF_ENGINE_PACKAGE.format(os="osx", version=TFF_ENGINE_VERSION),
)

java_export(
    name = "engine_package_linux",
    exports = [":engine"],
    maven_coordinates = TFF_ENGINE_PACKAGE.format(os="linux", version=TFF_ENGINE_VERSION),
)

alias(
    name = "publish",
    actual = select({
        "@platforms//os:macos": ":engine_package_osx.publish",
        "@platforms//os:linux": ":engine_package_linux.publish",
    }),
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

alias(
    name = "copy_aggregation_jni_to_resources",
    actual = select({
        "@platforms//os:macos": ":copy_aggregation_jni_to_resources_dylib",
        "@platforms//os:linux": ":copy_aggregation_jni_to_resources_so",
    }),
)

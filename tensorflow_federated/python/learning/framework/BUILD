load("//tensorflow_federated/tools:build_defs.bzl", "py_cpu_gpu_test")
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = [
    ":framework_packages",
    "//tensorflow_federated/python/learning:learning_users",
    "//tensorflow_federated/python/learning/algorithms:algorithms_packages",
    "//tensorflow_federated/python/learning/reconstruction:reconstruction_packages",
    "//tensorflow_federated/python/learning/templates:templates_packages",

    # TODO(b/151441025): This visibility is temporary and can be removed once
    # the circular dependency between `learning` and `framework` is removed.
    "//tensorflow_federated/python/learning:__pkg__",
])

package_group(
    name = "framework_packages",
    packages = ["//tensorflow_federated/python/learning/framework/..."],
)

licenses(["notice"])

py_library(
    name = "framework",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [":optimizer_utils"],
)

py_library(
    name = "optimizer_utils",
    srcs = ["optimizer_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/models:model_weights",
    ],
)

py_cpu_gpu_test(
    name = "optimizer_utils_test",
    srcs = ["optimizer_utils_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        ":optimizer_utils",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/models:model_weights",
    ],
)

load("//tensorflow_federated/tools:build_defs.bzl", "py_cpu_gpu_test")
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = [
    ":algorithms_packages",
    "//tensorflow_federated/python/learning:learning_visibility",
])

package_group(
    name = "algorithms_packages",
    packages = ["//tensorflow_federated/python/learning/algorithms/..."],
)

licenses(["notice"])

py_library(
    name = "client_scheduled_federated_averaging",
    srcs = ["client_scheduled_federated_averaging.py"],
    srcs_version = "PY3",
    deps = [
        ":example_weighted_federated_averaging",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:learning_process",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/framework:client_works",
        "//tensorflow_federated/python/learning/framework:composers",
        "//tensorflow_federated/python/learning/framework:distributors",
        "//tensorflow_federated/python/learning/framework:finalizers",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_cpu_gpu_test(
    name = "client_scheduled_federated_averaging_test",
    timeout = "long",
    srcs = ["client_scheduled_federated_averaging_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        ":client_scheduled_federated_averaging",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/learning:model_examples",
        "//tensorflow_federated/python/learning:model_update_aggregator",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "example_weighted_federated_averaging",
    srcs = ["example_weighted_federated_averaging.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:learning_process",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/framework:client_works",
        "//tensorflow_federated/python/learning/framework:composers",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
        "//tensorflow_federated/python/learning/framework:distributors",
        "//tensorflow_federated/python/learning/framework:finalizers",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_cpu_gpu_test(
    name = "example_weighted_federated_averaging_test",
    timeout = "long",
    srcs = ["example_weighted_federated_averaging_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        ":example_weighted_federated_averaging",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/learning:model_examples",
        "//tensorflow_federated/python/learning:model_update_aggregator",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

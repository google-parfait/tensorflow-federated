load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//tools:build_defs.bzl", "py_cpu_gpu_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":learning_packages",
        ":learning_users",
    ],
)

package_group(
    name = "learning_packages",
    packages = [
        "//tensorflow_federated/python/learning/...",
    ],
)

package_group(
    name = "learning_users",
    includes = ["//tensorflow_federated/python/simulation:simulation_packages"],
)

licenses(["notice"])

py_library(
    name = "learning",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":client_weight_lib",
        ":debug_measurements",
        ":model_update_aggregator",
        "//tensorflow_federated/python/learning/algorithms",
        "//tensorflow_federated/python/learning/metrics",
        "//tensorflow_federated/python/learning/models",
        "//tensorflow_federated/python/learning/optimizers",
        "//tensorflow_federated/python/learning/programs",
        "//tensorflow_federated/python/learning/templates",
    ],
)

py_library(
    name = "client_weight_lib",
    srcs = ["client_weight_lib.py"],
)

py_library(
    name = "loop_builder",
    srcs = ["loop_builder.py"],
)

py_cpu_gpu_test(
    name = "loop_builder_test",
    srcs = ["loop_builder_test.py"],
    deps = [":loop_builder"],
)

py_library(
    name = "model_update_aggregator",
    srcs = ["model_update_aggregator.py"],
    deps = [
        "//tensorflow_federated/python/aggregators:differential_privacy",
        "//tensorflow_federated/python/aggregators:distributed_dp",
        "//tensorflow_federated/python/aggregators:encoded",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/aggregators:quantile_estimation",
        "//tensorflow_federated/python/aggregators:robust",
        "//tensorflow_federated/python/aggregators:secure",
    ],
)

py_test(
    name = "model_update_aggregator_test",
    timeout = "long",
    srcs = ["model_update_aggregator_test.py"],
    deps = [
        ":debug_measurements",
        ":model_update_aggregator",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/core/backends/mapreduce:form_utils",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/core/test:static_assert",
    ],
)

py_library(
    name = "debug_measurements",
    srcs = ["debug_measurements.py"],
    deps = [
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:measurements",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "debug_measurements_test",
    srcs = ["debug_measurements_test.py"],
    deps = [
        ":debug_measurements",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "tensor_utils",
    srcs = ["tensor_utils.py"],
    deps = ["//tensorflow_federated/python/common_libs:py_typecheck"],
)

py_test(
    name = "tensor_utils_test",
    size = "small",
    srcs = ["tensor_utils_test.py"],
    deps = [":tensor_utils"],
)

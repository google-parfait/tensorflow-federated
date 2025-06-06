load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":metrics_packages",
        "//tensorflow_federated/python/learning:learning_users",
        "//tensorflow_federated/python/learning/algorithms:algorithms_packages",
        "//tensorflow_federated/python/learning/models:models_packages",
        "//tensorflow_federated/python/learning/templates:templates_packages",
    ],
)

package_group(
    name = "metrics_packages",
    packages = ["//tensorflow_federated/python/learning/metrics/..."],
)

licenses(["notice"])

py_library(
    name = "metrics",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [
        ":aggregator",
        ":counters",
        ":keras_finalizer",
        ":keras_utils",
        ":sampling_aggregation_factory",
        ":sum_aggregation_factory",
        ":types",
    ],
)

py_library(
    name = "sum_aggregation_factory",
    srcs = ["sum_aggregation_factory.py"],
    deps = [
        ":aggregation_utils",
        ":types",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:quantile_estimation",
        "//tensorflow_federated/python/aggregators:secure",
        "//tensorflow_federated/python/aggregators:sum_factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:estimation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "sum_aggregation_factory_test",
    srcs = ["sum_aggregation_factory_test.py"],
    shard_count = 2,
    deps = [
        ":sum_aggregation_factory",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:quantile_estimation",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:estimation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "aggregation_utils",
    srcs = ["aggregation_utils.py"],
    deps = [
        ":types",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "aggregation_utils_test",
    srcs = ["aggregation_utils_test.py"],
    deps = [
        ":aggregation_utils",
        ":keras_finalizer",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "aggregator",
    srcs = ["aggregator.py"],
    deps = [
        ":aggregation_utils",
        ":sampling_aggregation_factory",
        ":sum_aggregation_factory",
        ":types",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "aggregator_test",
    srcs = ["aggregator_test.py"],
    deps = [
        ":aggregator",
        ":keras_finalizer",
        ":sum_aggregation_factory",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "counters",
    srcs = ["counters.py"],
)

py_test(
    name = "counters_test",
    srcs = ["counters_test.py"],
    deps = [":counters"],
)

py_library(
    name = "keras_finalizer",
    srcs = ["keras_finalizer.py"],
    deps = ["//tensorflow_federated/python/common_libs:py_typecheck"],
)

py_test(
    name = "keras_finalizer_test",
    srcs = ["keras_finalizer_test.py"],
    deps = [
        ":keras_finalizer",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "keras_utils",
    srcs = ["keras_utils.py"],
    deps = ["//tensorflow_federated/python/core/environments/tensorflow_frontend:variable_utils"],
)

py_test(
    name = "keras_utils_test",
    srcs = ["keras_utils_test.py"],
    deps = [
        ":counters",
        ":keras_utils",
    ],
)

py_test(
    name = "sampling_aggregation_factory_test",
    srcs = ["sampling_aggregation_factory_test.py"],
    deps = [
        ":sampling_aggregation_factory",
        ":types",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    deps = ["@federated_language//federated_language"],
)

py_library(
    name = "sampling_aggregation_factory",
    srcs = ["sampling_aggregation_factory.py"],
    deps = [
        ":aggregation_utils",
        ":types",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:sampling",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

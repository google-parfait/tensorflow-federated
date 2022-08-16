load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":metrics_packages",
    "//tensorflow_federated/python/learning:learning_users",
    "//tensorflow_federated/python/learning/algorithms:algorithms_packages",
    "//tensorflow_federated/python/learning/framework:framework_packages",
    "//tensorflow_federated/python/learning/models:models_packages",
    "//tensorflow_federated/python/learning/reconstruction:reconstruction_packages",
    "//tensorflow_federated/python/learning/templates:templates_packages",

    # TODO(b/151441025): This visibility is temporary and can be removed once
    # the circular dependency between `learning` and `metrics` is removed.
    "//tensorflow_federated/python/learning:__pkg__",
])

package_group(
    name = "metrics_packages",
    packages = ["//tensorflow_federated/python/learning/metrics/..."],
)

licenses(["notice"])

py_library(
    name = "metrics",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [
        ":aggregation_factory",
        ":aggregator",
        ":counters",
        ":finalizer",
    ],
)

py_library(
    name = "aggregator",
    srcs = ["aggregator.py"],
    srcs_version = "PY3",
    deps = [
        ":aggregation_factory",
        ":aggregation_utils",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/learning/models:functional",
    ],
)

py_test(
    name = "aggregator_test",
    srcs = ["aggregator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":aggregation_factory",
        ":aggregator",
        ":finalizer",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/test:static_assert",
    ],
)

py_library(
    name = "counters",
    srcs = ["counters.py"],
    srcs_version = "PY3",
)

py_test(
    name = "counters_test",
    srcs = ["counters_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":counters"],
)

py_library(
    name = "finalizer",
    srcs = ["finalizer.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/common_libs:py_typecheck"],
)

py_test(
    name = "finalizer_test",
    srcs = ["finalizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":finalizer",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)

py_library(
    name = "keras_utils",
    srcs = ["keras_utils.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/tensorflow_libs:variable_utils"],
)

py_test(
    name = "keras_utils_test",
    srcs = ["keras_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":counters",
        ":keras_utils",
    ],
)

py_library(
    name = "aggregation_factory",
    srcs = ["aggregation_factory.py"],
    srcs_version = "PY3",
    deps = [
        ":aggregation_utils",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:quantile_estimation",
        "//tensorflow_federated/python/aggregators:secure",
        "//tensorflow_federated/python/aggregators:sum_factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:estimation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/learning/models:functional",
    ],
)

py_test(
    name = "aggregation_factory_test",
    srcs = ["aggregation_factory_test.py"],
    python_version = "PY3",
    shard_count = 2,
    srcs_version = "PY3",
    deps = [
        ":aggregation_factory",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:quantile_estimation",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:estimation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/core/test:static_assert",
    ],
)

py_library(
    name = "aggregation_utils",
    srcs = ["aggregation_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/learning/models:functional",
    ],
)

py_test(
    name = "aggregation_utils_test",
    srcs = ["aggregation_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":aggregation_utils",
        ":finalizer",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)

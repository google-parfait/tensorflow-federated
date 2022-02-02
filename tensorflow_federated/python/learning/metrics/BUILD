load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    "//tensorflow_federated/python/learning:learning_packages",
    "//tensorflow_federated/python/learning:learning_visibility",
])

licenses(["notice"])

py_library(
    name = "metrics",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [
        ":aggregator",
        ":counters",
        ":finalizer",
    ],
)

py_library(
    name = "aggregator",
    srcs = ["aggregator.py"],
    deps = [
        "//tensorflow_federated/python/aggregators:secure",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:model",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "aggregator_test",
    srcs = ["aggregator_test.py"],
    deps = [
        ":aggregator",
        ":finalizer",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/test:static_assert",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "counters",
    srcs = ["counters.py"],
    deps = ["@org_tensorflow//tensorflow:tensorflow_py"],
)

py_test(
    name = "counters_test",
    srcs = ["counters_test.py"],
    deps = [
        ":counters",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "finalizer",
    srcs = ["finalizer.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "finalizer_test",
    srcs = ["finalizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":finalizer",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

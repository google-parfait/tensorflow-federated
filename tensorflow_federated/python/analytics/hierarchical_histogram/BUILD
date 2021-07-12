load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//tensorflow_federated/python/analytics:__subpackages__"])

licenses(["notice"])

py_library(
    name = "hierarchical_histogram_factory",
    srcs = ["hierarchical_histogram_factory.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/aggregators:differential_privacy"],
)

py_test(
    name = "hierarchical_histogram_factory_test",
    srcs = ["hierarchical_histogram_factory_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":hierarchical_histogram_factory",
        "//tensorflow_federated/python/aggregators:differential_privacy",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_library(
    name = "hierarchical_histogram",
    srcs = ["hierarchical_histogram.py"],
    srcs_version = "PY3",
    deps = [
        ":hierarchical_histogram_factory",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "hierarchical_histogram_test",
    srcs = ["hierarchical_histogram_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":hierarchical_histogram",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
    ],
)

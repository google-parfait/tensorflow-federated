load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//tensorflow_federated/python/analytics:__subpackages__"])

licenses(["notice"])

py_library(
    name = "hierarchical_histogram_factory",
    srcs = ["hierarchical_histogram_factory.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/aggregators:differential_privacy",
        "//tensorflow_federated/python/aggregators:secure",
        "//tensorflow_federated/python/aggregators:sum_factory",
    ],
)

py_test(
    name = "hierarchical_histogram_factory_test",
    srcs = ["hierarchical_histogram_factory_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":build_tree_from_leaf",
        ":hierarchical_histogram_factory",
        "//tensorflow_federated/python/aggregators:differential_privacy",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
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
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
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
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
    ],
)

py_library(
    name = "hierarchical_histogram_decoder",
    srcs = ["hierarchical_histogram_decoder.py"],
    srcs_version = "PY3",
    deps = [":build_tree_from_leaf"],
)

py_test(
    name = "hierarchical_histogram_decoder_test",
    srcs = ["hierarchical_histogram_decoder_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":build_tree_from_leaf",
        ":hierarchical_histogram_decoder",
    ],
)

py_library(
    name = "build_tree_from_leaf",
    srcs = ["build_tree_from_leaf.py"],
    srcs_version = "PY3",
)

py_test(
    name = "build_tree_from_leaf_test",
    srcs = ["build_tree_from_leaf_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":build_tree_from_leaf"],
)

load("@pip//:requirements.bzl", "requirement")
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
        ":finalizer",
    ],
)

py_test(
    name = "finalizer_test",
    srcs = ["finalizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        ":finalizer",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)

py_test(
    name = "aggregator_test",
    srcs = ["aggregator_test.py"],
    deps = [
        requirement("tf-nightly"),
        ":aggregator",
        ":finalizer",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)

py_library(
    name = "finalizer",
    srcs = ["finalizer.py"],
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_library(
    name = "aggregator",
    srcs = ["aggregator.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/learning:model",
    ],
)

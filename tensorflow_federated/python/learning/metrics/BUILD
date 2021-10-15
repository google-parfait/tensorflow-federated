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
    deps = [":finalizer"],
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

py_library(
    name = "finalizer",
    srcs = ["finalizer.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

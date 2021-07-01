load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/learning:learning_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/learning/optimizers/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "optimizers",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "keras_optimizer",
    srcs = ["keras_optimizer.py"],
    srcs_version = "PY3",
    deps = [":optimizer"],
)

py_test(
    name = "keras_optimizer_test",
    srcs = ["keras_optimizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_optimizer",
        ":optimizer_test_utils",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
    ],
)

py_library(
    name = "optimizer",
    srcs = ["optimizer.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/common_libs:py_typecheck"],
)

py_test(
    name = "optimizer_test",
    srcs = ["optimizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)

py_library(
    name = "optimizer_test_utils",
    testonly = True,
    srcs = ["optimizer_test_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "optimizer_test_utils_test",
    srcs = ["optimizer_test_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":optimizer_test_utils",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)

py_library(
    name = "sgdm",
    srcs = ["sgdm.py"],
    srcs_version = "PY3",
    deps = [":optimizer"],
)

py_test(
    name = "sgdm_test",
    srcs = ["sgdm_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":optimizer_test_utils",
        ":sgdm",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)

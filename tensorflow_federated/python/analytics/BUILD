load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    packages = [
        "//tensorflow_federated/python/analytics/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "analytics",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":data_processing",
        ":histogram_processing",
    ],
)

py_library(
    name = "data_processing",
    srcs = ["data_processing.py"],
    srcs_version = "PY3",
)

py_test(
    name = "data_processing_test",
    srcs = ["data_processing_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":data_processing"],
)

py_library(
    name = "histogram_test_utils",
    testonly = True,
    srcs = ["histogram_test_utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "histogram_processing",
    srcs = ["histogram_processing.py"],
    srcs_version = "PY3",
)

py_test(
    name = "histogram_processing_test",
    srcs = ["histogram_processing_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":histogram_processing",
        ":histogram_test_utils",
    ],
)

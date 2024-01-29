load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":tensorflow_libs_packages",
        "//tensorflow_federated/python:internal_packages",
    ],
)

package_group(
    name = "tensorflow_libs_packages",
    packages = ["//tensorflow_federated/python/tensorflow_libs..."],
)

licenses(["notice"])

py_library(
    name = "tensorflow_libs",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "graph_optimizations",
    srcs = ["graph_optimizations.py"],
    deps = [":graph_spec"],
)

py_test(
    name = "graph_optimizations_test",
    srcs = ["graph_optimizations_test.py"],
    deps = [
        ":graph_optimizations",
        ":graph_spec",
    ],
)

py_library(
    name = "graph_spec",
    srcs = ["graph_spec.py"],
)

py_test(
    name = "graph_spec_test",
    srcs = ["graph_spec_test.py"],
    deps = [":graph_spec"],
)

py_library(
    name = "graph_utils",
    srcs = ["graph_utils.py"],
)

py_test(
    name = "graph_utils_test",
    srcs = ["graph_utils_test.py"],
    deps = [":graph_utils"],
)

py_library(
    name = "serialization_utils",
    srcs = ["serialization_utils.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "serialization_utils_test",
    size = "small",
    srcs = ["serialization_utils_test.py"],
    deps = [
        ":serialization_utils",
        "@com_google_protobuf//:protobuf_python",
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

py_library(
    name = "tensorflow_test_utils",
    testonly = True,
    srcs = ["tensorflow_test_utils.py"],
)

py_library(
    name = "variable_utils",
    srcs = ["variable_utils.py"],
)

py_test(
    name = "variable_utils_test",
    size = "small",
    timeout = "moderate",
    srcs = ["variable_utils_test.py"],
    deps = [":variable_utils"],
)

py_library(
    name = "version_check",
    srcs = ["version_check.py"],
)

py_test(
    name = "version_check_test",
    size = "small",
    srcs = ["version_check_test.py"],
    deps = [":version_check"],
)

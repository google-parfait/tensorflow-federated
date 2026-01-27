load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//tensorflow_federated:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "query",
    srcs = ["query.py"],
)

py_test(
    name = "query_test",
    srcs = ["query_test.py"],
    deps = [
        ":query",
        ":test_utils",
    ],
)

py_library(
    name = "tree",
    srcs = ["tree.py"],
    deps = [":query"],
)

py_test(
    name = "tree_test",
    srcs = ["tree_test.py"],
    deps = [
        ":test_utils",
        ":tree",
    ],
)

py_library(
    name = "quantile",
    srcs = ["quantile.py"],
    deps = [
        ":query",
        ":tree",
    ],
)

py_test(
    name = "quantile_test",
    srcs = ["quantile_test.py"],
    deps = [
        ":quantile",
        ":test_utils",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
)

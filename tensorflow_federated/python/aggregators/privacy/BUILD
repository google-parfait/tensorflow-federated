load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//tensorflow_federated:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "query",
    srcs = ["query.py"],
    deps = [
        "@pypi//dp_accounting",
        "@pypi//tensorflow",
    ],
)

py_test(
    name = "query_test",
    srcs = ["query_test.py"],
    deps = [
        ":query",
        ":test_utils",
        "@pypi//absl_py",
        "@pypi//numpy",
        "@pypi//tensorflow",
    ],
)

py_library(
    name = "tree",
    srcs = ["tree.py"],
    deps = [
        ":query",
        "@pypi//dp_accounting",
        "@pypi//tensorflow",
    ],
)

py_test(
    name = "tree_test",
    srcs = ["tree_test.py"],
    deps = [
        ":test_utils",
        ":tree",
        "@pypi//absl_py",
        "@pypi//numpy",
        "@pypi//tensorflow",
    ],
)

py_library(
    name = "quantile",
    srcs = ["quantile.py"],
    deps = [
        ":query",
        ":tree",
        "@pypi//dp_accounting",
        "@pypi//tensorflow",
    ],
)

py_test(
    name = "quantile_test",
    srcs = ["quantile_test.py"],
    deps = [
        ":quantile",
        ":test_utils",
        "@pypi//absl_py",
        "@pypi//numpy",
        "@pypi//tensorflow",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
)

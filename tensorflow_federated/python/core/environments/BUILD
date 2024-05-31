load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

package_group(
    name = "environments_packages",
    packages = [
        "//tensorflow_federated/python/core/environments/...",
    ],
)

package_group(
    name = "environments_users",
    includes = [
        "//tensorflow_federated/python/core:core_users",
        "//tensorflow_federated/python/core/backends:backends_packages",
    ],
    packages = [
        "//tensorflow_federated",
    ],
)

licenses(["notice"])

py_library(
    name = "environments",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

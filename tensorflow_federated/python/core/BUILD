load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

package_group(
    name = "core_packages",
    packages = ["//tensorflow_federated/python/core/..."],
)

package_group(
    name = "core_users",
    includes = [
        "//tensorflow_federated/python/aggregators:aggregators_packages",
        "//tensorflow_federated/python/analytics:analytics_packages",
        "//tensorflow_federated/python/learning:learning_packages",
        "//tensorflow_federated/python/program:program_packages",
        "//tensorflow_federated/python/simulation:simulation_packages",
    ],
)

licenses(["notice"])

py_library(
    name = "core",
    srcs = ["__init__.py"],
    visibility = ["//tools/python_package:python_package_tool"],
)

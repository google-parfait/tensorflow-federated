load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = [
    ":program_packages",
    "//tensorflow_federated/python/analytics:analytics_packages",
    "//tensorflow_federated/python/learning:learning_packages",
    "//tensorflow_federated/python/simulation:simulation_packages",
])

package_group(
    name = "program_packages",
    packages = ["//tensorflow_federated/python/program/..."],
)

licenses(["notice"])

py_library(
    name = "program",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated:__pkg__"],
)

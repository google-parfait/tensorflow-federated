load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/simulation:simulation_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/simulation/models/...",
        "//tensorflow_federated/python/simulation/baselines/cifar100/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "models",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/simulation:__pkg__"],
    deps = [":mnist"],
)

py_library(
    name = "group_norm",
    srcs = ["group_norm.py"],
    srcs_version = "PY3",
)

py_test(
    name = "group_norm_test",
    timeout = "moderate",
    srcs = ["group_norm_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":group_norm"],
)

py_library(
    name = "mnist",
    srcs = ["mnist.py"],
    srcs_version = "PY3",
)

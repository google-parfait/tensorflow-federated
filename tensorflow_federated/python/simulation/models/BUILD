load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":models_packages",
    "//tensorflow_federated/python/simulation:simulation_visibility",
    "//tensorflow_federated/python/simulation/baselines:baselines_packages",
])

package_group(
    name = "models_packages",
    packages = ["//tensorflow_federated/python/simulation/models/..."],
)

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
    deps = [
        ":group_norm",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "mnist",
    srcs = ["mnist.py"],
    srcs_version = "PY3",
)

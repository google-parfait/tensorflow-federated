load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":models_packages",
        "//tensorflow_federated/python/simulation:simulation_users",
        "//tensorflow_federated/python/simulation/baselines:baselines_packages",
    ],
)

package_group(
    name = "models_packages",
    packages = ["//tensorflow_federated/python/simulation/models/..."],
)

licenses(["notice"])

py_library(
    name = "models",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/simulation:__pkg__"],
    deps = [":mnist"],
)

py_library(
    name = "mnist",
    srcs = ["mnist.py"],
    deps = [
        "//tensorflow_federated/python/learning/metrics:counters",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "mobilenet_v2",
    srcs = ["mobilenet_v2.py"],
    deps = [
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "mobilenet_v2_test",
    srcs = ["mobilenet_v2_test.py"],
    deps = [
        ":mobilenet_v2",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "resnet_models",
    srcs = ["resnet_models.py"],
    deps = [
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "resnet_models_test",
    srcs = ["resnet_models_test.py"],
    deps = [
        ":resnet_models",
        "@pypi_tensorflow//:pkg",
    ],
)

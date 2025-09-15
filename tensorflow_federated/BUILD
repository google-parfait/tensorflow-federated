load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_library(
    name = "tensorflow_federated",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":version",
        "//tensorflow_federated/python/aggregators",
        "//tensorflow_federated/python/analytics",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/backends",
        "//tensorflow_federated/python/core/environments/tensorflow",
        "//tensorflow_federated/python/core/framework",
        "//tensorflow_federated/python/core/templates",
        "//tensorflow_federated/python/learning",
        "//tensorflow_federated/python/program",
        "//tensorflow_federated/python/simulation",
        "@pypi_absl_py//:pkg",
        "@pypi_attrs//:pkg",
        "@pypi_cachetools//:pkg",
        "@pypi_dm_tree//:pkg",
        "@pypi_dp_accounting//:pkg",
        "@pypi_google_vizier//:pkg",
        "@pypi_googleapis_common_protos//:pkg",
        "@pypi_grpcio//:pkg",
        "@pypi_jax//:pkg",
        "@pypi_jaxlib//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_portpicker//:pkg",
        "@pypi_scipy//:pkg",
        "@pypi_tensorflow_model_optimization//:pkg",
        "@pypi_tensorflow_privacy//:pkg",
        "@pypi_tensorflow//:pkg",
        "@pypi_tqdm//:pkg",
        "@pypi_typing_extensions//:pkg",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
)

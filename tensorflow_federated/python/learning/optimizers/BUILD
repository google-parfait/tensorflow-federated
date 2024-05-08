load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":optimizers_packages",
        "//tensorflow_federated/python/learning:learning_users",
        "//tensorflow_federated/python/learning/algorithms:algorithms_packages",
        "//tensorflow_federated/python/learning/templates:templates_packages",
    ],
)

package_group(
    name = "optimizers_packages",
    packages = ["//tensorflow_federated/python/learning/optimizers/..."],
)

licenses(["notice"])

py_library(
    name = "adafactor",
    srcs = ["adafactor.py"],
    deps = [
        ":optimizer",
        "//tensorflow_federated/python/common_libs:structure",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "adafactor_test",
    srcs = ["adafactor_test.py"],
    deps = [
        ":adafactor",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "adagrad",
    srcs = ["adagrad.py"],
    deps = [
        ":optimizer",
        "//tensorflow_federated/python/common_libs:structure",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "adagrad_test",
    srcs = ["adagrad_test.py"],
    deps = [
        ":adagrad",
        ":optimizer",
        ":optimizer_test_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "adam",
    srcs = ["adam.py"],
    deps = [
        ":optimizer",
        "//tensorflow_federated/python/common_libs:structure",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "adam_test",
    srcs = ["adam_test.py"],
    deps = [
        ":adam",
        ":optimizer",
        ":optimizer_test_utils",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "optimizers",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [
        ":adafactor",
        ":adagrad",
        ":adam",
        ":optimizer",
        ":rmsprop",
        ":scheduling",
        ":sgdm",
        ":yogi",
    ],
)

py_library(
    name = "keras_optimizer",
    srcs = ["keras_optimizer.py"],
    deps = [
        ":optimizer",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "keras_optimizer_test",
    srcs = ["keras_optimizer_test.py"],
    deps = [
        ":keras_optimizer",
        ":optimizer",
        ":optimizer_test_utils",
        ":sgdm",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "integration_test",
    srcs = ["integration_test.py"],
    deps = [
        ":adagrad",
        ":adam",
        ":rmsprop",
        ":scheduling",
        ":sgdm",
        ":yogi",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "optimizer",
    srcs = ["optimizer.py"],
    deps = [
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "optimizer_test",
    srcs = ["optimizer_test.py"],
    deps = [
        ":optimizer",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "optimizer_test_utils",
    testonly = True,
    srcs = ["optimizer_test_utils.py"],
    deps = [
        ":optimizer",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "optimizer_test_utils_test",
    srcs = ["optimizer_test_utils_test.py"],
    deps = [
        ":optimizer_test_utils",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "rmsprop",
    srcs = ["rmsprop.py"],
    deps = [
        ":optimizer",
        "//tensorflow_federated/python/common_libs:structure",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "rmsprop_test",
    srcs = ["rmsprop_test.py"],
    deps = [
        ":optimizer",
        ":optimizer_test_utils",
        ":rmsprop",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "scheduling",
    srcs = ["scheduling.py"],
    deps = [
        ":optimizer",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "scheduling_test",
    srcs = ["scheduling_test.py"],
    deps = [
        ":adagrad",
        ":adam",
        ":optimizer",
        ":rmsprop",
        ":scheduling",
        ":sgdm",
        ":yogi",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "sgdm",
    srcs = ["sgdm.py"],
    deps = [
        ":optimizer",
        "//tensorflow_federated/python/common_libs:structure",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "sgdm_test",
    srcs = ["sgdm_test.py"],
    deps = [
        ":optimizer",
        ":optimizer_test_utils",
        ":sgdm",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

py_library(
    name = "yogi",
    srcs = ["yogi.py"],
    deps = [
        ":optimizer",
        "//tensorflow_federated/python/common_libs:structure",
        "@pypi_tensorflow//:pkg",
    ],
)

py_test(
    name = "yogi_test",
    srcs = ["yogi_test.py"],
    deps = [
        ":optimizer",
        ":optimizer_test_utils",
        ":yogi",
        "@pypi_absl_py//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

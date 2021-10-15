load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":optimizers_packages",
    "//tensorflow_federated/python/learning:learning_visibility",
    "//tensorflow_federated/python/learning/algorithms:algorithms_packages",
    "//tensorflow_federated/python/learning/reconstruction:reconstruction_packages",

    # TODO(b/151441025): This visibility is temporary and can be removed once
    # the dependencies between `tff.learning` and `tff.learning.framework` are
    # cleaned up.
    "//tensorflow_federated/python/learning/framework:framework_packages",
    "//tensorflow_federated/python/learning:__pkg__",
])

package_group(
    name = "optimizers_packages",
    packages = ["//tensorflow_federated/python/learning/optimizers/..."],
)

licenses(["notice"])

py_library(
    name = "optimizers",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [
        ":optimizer",
        ":sgdm",
    ],
)

py_library(
    name = "keras_optimizer",
    srcs = ["keras_optimizer.py"],
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "keras_optimizer_test",
    srcs = ["keras_optimizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_optimizer",
        ":optimizer",
        ":optimizer_test_utils",
        ":sgdm",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "optimizer",
    srcs = ["optimizer.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "optimizer_test",
    srcs = ["optimizer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        "//tensorflow_federated/python/core/api:test_case",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "optimizer_test_utils",
    testonly = True,
    srcs = ["optimizer_test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        "//tensorflow_federated/python/core/api:test_case",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "optimizer_test_utils_test",
    srcs = ["optimizer_test_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":optimizer_test_utils",
        "//tensorflow_federated/python/core/api:test_case",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "sgdm",
    srcs = ["sgdm.py"],
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "sgdm_test",
    srcs = ["sgdm_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":optimizer",
        ":optimizer_test_utils",
        ":sgdm",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

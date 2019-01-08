# Placeholder for internal Python version compatibility macro.

package(default_visibility = ["//tensorflow_federated/python/learning:__subpackages__"])

licenses(["notice"])  # Apache 2.0

# This is the import for clients that contribute components to federated learning (see `../README.md`).
py_library(
    name = "framework",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":optimizer_utils",
        "//tensorflow_federated/python/learning:federated_averaging",
        "//tensorflow_federated/python/learning:model_utils",
    ],
)

py_library(
    name = "optimizer_utils",
    srcs = ["optimizer_utils.py"],
    deps = [
        "//tensorflow_federated/python/core",
        "//tensorflow_federated/python/learning:model_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "optimizer_utils_test",
    srcs = ["optimizer_utils_test.py"],
    default_python_version = "PY3",
    deps = [
        ":optimizer_utils",
        "//tensorflow_federated/python/common_libs:test_utils",
        "//tensorflow_federated/python/learning:model_examples",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

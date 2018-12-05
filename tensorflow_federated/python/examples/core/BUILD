# Placeholder for internal Python version compatibility macro.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "training",
    srcs = ["training.py"],
    deps = [
        "//tensorflow_federated/python/core/api",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "training_test",
    size = "small",
    srcs = ["training_test.py"],
    deps = [
        ":training",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "landmark_preprocessing",
    srcs = ["landmark_preprocessing.py"],
    deps = ["//tensorflow_federated/python/simulation/baselines:client_spec"],
)

py_test(
    name = "landmark_preprocessing_test",
    srcs = ["landmark_preprocessing_test.py"],
    deps = [
        ":landmark_preprocessing",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/simulation/baselines:client_spec",
    ],
)

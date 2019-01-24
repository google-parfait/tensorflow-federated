# Package for TensorFlow containing tools for lossy compression of models.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "encoders",
    srcs = ["encoders.py"],
    deps = [
        "//tensorflow_federated/python/tensorflow_libs/model_compression/core:encoding_stage",
        "@six_archive//:six",
    ],
)

py_test(
    name = "encoders_test",
    size = "small",
    srcs = ["encoders_test.py"],
    deps = [
        ":encoders",
        "//tensorflow_federated/python/tensorflow_libs/model_compression/core:test_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

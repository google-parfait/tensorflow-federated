# Package for TensorFlow containing tools for lossy compression of models.

# Placeholder for internal Python version compatibility macro.

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
    srcs = ["encoders_test.py"],
    default_python_version = "PY3",
    deps = [
        ":encoders",
        "//tensorflow_federated/python/tensorflow_libs/model_compression/core:test_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

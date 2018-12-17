# Placeholder for internal Python version compatibility macro.

# Eventually we will expose symbols from learning/ that can be used externally
# via a public __init__.py file.
package(default_visibility = ["//tensorflow_federated/python/learning:__subpackages__"])

licenses(["notice"])  # Apache 2.0

# This is the import for clients that use federated learning (see `README.md`).
py_library(
    name = "learning",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":model",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = ["@six_archive//:six"],
)

py_library(
    name = "model_examples",
    srcs = ["model_examples.py"],
    deps = [
        ":model",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "model_examples_test",
    srcs = ["model_examples_test.py"],
    deps = [
        ":model_examples",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

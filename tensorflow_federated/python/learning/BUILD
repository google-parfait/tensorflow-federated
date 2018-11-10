# Eventually we will expose symbols from learning/ that can be used externally
# via a public __init__.py file.
package(default_visibility = ["//tensorflow_federated/python/learning:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "model_fn",
    srcs = ["model_fn.py"],
    deps = [
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_fn_examples",
    srcs = ["model_fn_examples.py"],
    deps = [
        ":model_fn",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "model_fn_test",
    srcs = ["model_fn_test.py"],
    deps = [
        ":model_fn",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "model_fn_examples_test",
    srcs = ["model_fn_examples_test.py"],
    deps = [
        ":model_fn",
        ":model_fn_examples",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

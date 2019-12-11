package(default_visibility = ["//tensorflow_federated/python/learning:__subpackages__"])

licenses(["notice"])

# This is the import for clients that use federated learning (see `README.md`).
py_library(
    name = "learning",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":federated_averaging",
        ":federated_evaluation",
        ":federated_sgd",
        ":keras_utils",
        ":model",
        "//tensorflow_federated/python/learning/framework",
        "//tensorflow_federated/python/learning/framework:optimizer_utils",
    ],
)

py_library(
    name = "federated_averaging",
    srcs = ["federated_averaging.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core",
        "//tensorflow_federated/python/learning/framework:optimizer_utils",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)

py_test(
    name = "federated_averaging_benchmark",
    srcs = ["federated_averaging_benchmark.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":federated_averaging",
        ":keras_utils",
        ":model_examples",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core",
    ],
)

py_test(
    name = "federated_averaging_test",
    size = "medium",
    srcs = ["federated_averaging_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":federated_averaging",
        ":keras_utils",
        ":model_examples",
        ":model_utils",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core",
    ],
)

py_library(
    name = "federated_evaluation",
    srcs = ["federated_evaluation.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model_utils",
        "//tensorflow_federated/python/core",
    ],
)

py_test(
    name = "federated_evaluation_test",
    srcs = ["federated_evaluation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":federated_evaluation",
        ":keras_utils",
        ":model",
        ":model_utils",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core",
    ],
)

py_library(
    name = "federated_sgd",
    srcs = ["federated_sgd.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model",
        ":model_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core",
        "//tensorflow_federated/python/learning/framework:optimizer_utils",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)

py_test(
    name = "federated_sgd_test",
    size = "large",
    srcs = ["federated_sgd_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":federated_sgd",
        ":keras_utils",
        ":model_examples",
        ":model_utils",
        "//tensorflow_federated/python/common_libs:test",
    ],
)

py_library(
    name = "keras_utils",
    srcs = ["keras_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model",
        ":model_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core",
        "//tensorflow_federated/python/tensorflow_libs:graph_keys",
    ],
)

py_test(
    name = "keras_utils_test",
    size = "medium",
    srcs = ["keras_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_utils",
        ":model_examples",
        ":model_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "model_examples",
    srcs = ["model_examples.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model",
        "//tensorflow_federated/python/core",
    ],
)

py_test(
    name = "model_examples_test",
    size = "small",
    srcs = ["model_examples_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_examples",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core",
    ],
)

py_library(
    name = "model_utils",
    srcs = ["model_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "model_utils_test",
    srcs = ["model_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_examples",
        ":model_utils",
        "//tensorflow_federated/python/common_libs:test",
    ],
)

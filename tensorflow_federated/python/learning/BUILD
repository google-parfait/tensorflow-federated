package(default_visibility = ["//tensorflow_federated/python/learning:__subpackages__"])

licenses(["notice"])  # Apache 2.0

# This is the import for clients that use federated learning (see `README.md`).
py_library(
    name = "learning",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":federated_averaging",
        ":federated_evaluation",
        ":model",
        ":model_utils",
        "//tensorflow_federated/python/learning/framework",
        "//tensorflow_federated/python/learning/framework:optimizer_utils",
    ],
)

py_library(
    name = "federated_averaging",
    srcs = ["federated_averaging.py"],
    deps = [
        ":model_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/learning/framework:optimizer_utils",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/keras/optimizer_v2",
    ],
)

py_test(
    name = "federated_averaging_benchmark",
    size = "medium",
    srcs = ["federated_averaging_benchmark.py"],
    deps = [
        ":federated_averaging",
        ":model_examples",
        ":model_utils",
        "//tensorflow_federated/python/core",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "federated_averaging_test",
    size = "large",
    srcs = ["federated_averaging_test.py"],
    deps = [
        ":federated_averaging",
        ":model_examples",
        ":model_utils",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core",
        "//tensorflow_federated/python/learning/framework:optimizer_utils",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/keras/optimizer_v2",
        "@six_archive//:six",
    ],
)

py_library(
    name = "federated_evaluation",
    srcs = ["federated_evaluation.py"],
    deps = [
        ":model_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "federated_evaluation_test",
    size = "medium",
    srcs = ["federated_evaluation_test.py"],
    deps = [
        ":federated_evaluation",
        ":model",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "federated_sgd",
    srcs = ["federated_sgd.py"],
    deps = [
        ":model",
        ":model_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/learning/framework:optimizer_utils",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "federated_sgd_test",
    size = "medium",
    srcs = ["federated_sgd_test.py"],
    deps = [
        ":federated_sgd",
        ":model_examples",
        ":model_utils",
        "//tensorflow_federated/python/common_libs:test",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
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
        "//tensorflow_federated/python/core",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "model_examples_test",
    size = "small",
    srcs = ["model_examples_test.py"],
    deps = [
        ":model_examples",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "model_utils",
    srcs = ["model_utils.py"],
    deps = [
        ":model",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core",
        "//tensorflow_federated/python/tensorflow_libs:graph_keys",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/keras:engine",
        "@six_archive//:six",
    ],
)

py_test(
    name = "model_utils_test",
    size = "large",
    srcs = ["model_utils_test.py"],
    # TODO(b/123578208): Remove deep keras.optimizer_v2 and keras.engine
    # imports after updating TF version.
    deps = [
        ":model_examples",
        ":model_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/keras:engine",
        "@org_tensorflow//tensorflow/python/keras/optimizer_v2",
        "@six_archive//:six",
    ],
)

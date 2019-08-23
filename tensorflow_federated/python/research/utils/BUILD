package(default_visibility = ["//tensorflow_federated/python/research:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "utils",
    srcs = ["__init__.py"],
    deps = [":utils_impl"],
)

py_library(
    name = "utils_impl",
    srcs = ["utils_impl.py"],
    srcs_version = "PY3",
)

py_test(
    name = "utils_impl_test",
    srcs = ["utils_impl_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":utils_impl"],
)

py_library(
    name = "training_loops",
    srcs = ["training_loops.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "training_loops_test",
    size = "large",
    srcs = ["training_loops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":training_loops",
        "//tensorflow_federated",
        "//tensorflow_federated/python/examples/mnist:models",
    ],
)

py_library(
    name = "checkpoint_utils",
    srcs = ["checkpoint_utils.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/common_libs:anonymous_tuple"],
)

py_test(
    name = "checkpoint_utils_test",
    size = "large",
    srcs = ["checkpoint_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":checkpoint_utils",
        "//tensorflow_federated",
        "//tensorflow_federated/python/examples/mnist:models",
    ],
)

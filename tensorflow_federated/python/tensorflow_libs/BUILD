package(default_visibility = ["//tensorflow_federated:__subpackages__"])

licenses(["notice"])

py_library(
    name = "tensorflow_libs",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "graph_keys",
    srcs = ["graph_keys.py"],
)

py_library(
    name = "graph_merge",
    srcs = ["graph_merge.py"],
)

py_test(
    name = "graph_merge_test",
    srcs = ["graph_merge_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":graph_merge",
        "//tensorflow_federated/python/common_libs:test",
    ],
)

py_library(
    name = "nest",
    srcs = ["nest.py"],
)

py_test(
    name = "nest_test",
    srcs = ["nest_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":nest"],
)

py_library(
    name = "tensor_utils",
    srcs = ["tensor_utils.py"],
    deps = ["//tensorflow_federated/python/common_libs:py_typecheck"],
)

py_test(
    name = "tensor_utils_test",
    size = "small",
    srcs = ["tensor_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensor_utils",
        "//tensorflow_federated/python/common_libs:test",
    ],
)

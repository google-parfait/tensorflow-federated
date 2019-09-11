package(default_visibility = [
    "//tensorflow_federated/python/learning:__subpackages__",
    "//tensorflow_federated/python/research:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

# This is the import for clients that contribute components to federated learning (see `../README.md`).
py_library(
    name = "framework",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":encoding_utils",
        ":optimizer_utils",
        "//tensorflow_federated/python/learning:model_utils",
    ],
)

py_library(
    name = "encoding_utils",
    srcs = ["encoding_utils.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core",
        "//tensorflow_federated/python/learning:model_utils",
    ],
)

py_test(
    name = "encoding_utils_test",
    srcs = ["encoding_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":encoding_utils",
        ":optimizer_utils",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core",
        "//tensorflow_federated/python/learning:model_examples",
    ],
)

py_library(
    name = "optimizer_utils",
    srcs = ["optimizer_utils.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)

py_test(
    name = "optimizer_utils_test",
    srcs = ["optimizer_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":optimizer_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core",
        "//tensorflow_federated/python/learning:model_examples",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)

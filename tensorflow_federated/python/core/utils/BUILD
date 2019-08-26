package(default_visibility = ["//tensorflow_federated/python/core:__subpackages__"])

licenses(["notice"])  # Apache 2.0

# This is the import for other components of TFF that are layered upon the Core
# utils.
py_library(
    name = "utils",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":computation_utils",
        ":encoding_utils",
        ":federated_aggregations",
        ":tf_computation_utils",
    ],
)

py_library(
    name = "computation_utils",
    srcs = ["computation_utils.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api",
    ],
)

py_test(
    name = "computation_utils_test",
    size = "small",
    srcs = ["computation_utils_test.py"],
    deps = [
        ":computation_utils",
        "//tensorflow_federated/python/core/api",
    ],
)

py_library(
    name = "encoding_utils",
    srcs = ["encoding_utils.py"],
    deps = [
        ":computation_utils",
        "//tensorflow_federated/python/core/api",
        "//tensorflow_federated/python/core/framework",
        "//tensorflow_federated/python/tensorflow_libs:nest",
    ],
)

py_test(
    name = "encoding_utils_test",
    size = "medium",
    srcs = ["encoding_utils_test.py"],
    deps = [
        ":computation_utils",
        ":encoding_utils",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api",
    ],
)

py_library(
    name = "federated_aggregations",
    srcs = ["federated_aggregations.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api",
    ],
)

py_test(
    name = "federated_aggregations_test",
    size = "small",
    srcs = ["federated_aggregations_test.py"],
    deps = [
        ":federated_aggregations",
        "//tensorflow_federated/python/core/api",
    ],
)

py_library(
    name = "test",
    srcs = ["test.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl:executor_stacks",
        "//tensorflow_federated/python/core/impl:set_default_executor",
    ],
)

py_library(
    name = "tf_computation_utils",
    srcs = ["tf_computation_utils.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_test(
    name = "tf_computation_utils_test",
    size = "small",
    srcs = ["tf_computation_utils_test.py"],
    deps = [
        ":tf_computation_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
    ],
)

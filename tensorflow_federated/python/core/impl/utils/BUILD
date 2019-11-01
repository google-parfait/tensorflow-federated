package(default_visibility = [
    "//tensorflow_federated/python/core/api:__subpackages__",
    "//tensorflow_federated/python/core/backends:__subpackages__",
    "//tensorflow_federated/python/core/framework:__subpackages__",
    "//tensorflow_federated/python/core/impl/compiler:__subpackages__",
    "//tensorflow_federated/python/core/impl/utils:__subpackages__",
    "//tensorflow_federated/python/core/impl/wrappers:__subpackages__",
    "//tensorflow_federated/python/core/utils:__subpackages__",
    # TODO(b/143772942): This package is temporary and can be removed once all
    # the appropriate  modules have been moved out of the `impl` package.
    "//tensorflow_federated/python/core/impl:__pkg__",
])

licenses(["notice"])

py_library(
    name = "utils",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "function_utils",
    srcs = ["function_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:value_base",
        "//tensorflow_federated/python/core/impl:context_base",
        "//tensorflow_federated/python/core/impl:context_stack_base",
        "//tensorflow_federated/python/core/impl:type_utils",
    ],
)

py_test(
    name = "function_utils_test",
    size = "small",
    srcs = ["function_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":function_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:context_base",
        "//tensorflow_federated/python/core/impl:context_stack_impl",
        "//tensorflow_federated/python/core/impl:type_utils",
    ],
)

py_library(
    name = "tensorflow_utils",
    srcs = ["tensorflow_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":function_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:type_utils",
    ],
)

py_test(
    name = "tensorflow_utils_test",
    size = "small",
    srcs = ["tensorflow_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tensorflow_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:type_utils",
    ],
)

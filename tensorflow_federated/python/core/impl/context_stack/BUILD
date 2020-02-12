package(default_visibility = [
    "//tensorflow_federated/python/core/api:__subpackages__",
    "//tensorflow_federated/python/core/backends:__subpackages__",
    "//tensorflow_federated/python/core/framework:__subpackages__",
    "//tensorflow_federated/python/core/impl/context_stack:__subpackages__",
    "//tensorflow_federated/python/core/impl/wrappers:__subpackages__",
    "//tensorflow_federated/python/core/templates:__subpackages__",
    "//tensorflow_federated/python/core/utils:__subpackages__",
    # TODO(b/143772942): This package is temporary and can be removed once all
    # the appropriate  modules have been moved out of the `impl` package.
    "//tensorflow_federated/python/core/impl:__pkg__",
])

licenses(["notice"])

py_library(
    name = "context_stack",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "context_stack_impl",
    srcs = ["context_stack_impl.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl:context_base",
        "//tensorflow_federated/python/core/impl:context_stack_base",
        "//tensorflow_federated/python/core/impl:execution_context",
        "//tensorflow_federated/python/core/impl:executor_stacks",
    ],
)

py_test(
    name = "context_stack_impl_test",
    size = "small",
    srcs = ["context_stack_impl_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context_stack_impl",
        "//tensorflow_federated/python/core/impl:context_stack_test_utils",
        "//tensorflow_federated/python/core/impl:execution_context",
    ],
)

py_library(
    name = "set_default_executor",
    srcs = ["set_default_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":context_stack_impl",
        "//tensorflow_federated/python/core/impl:execution_context",
        "//tensorflow_federated/python/core/impl:reference_executor",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
    ],
)

py_test(
    name = "set_default_executor_test",
    size = "small",
    srcs = ["set_default_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context_stack_impl",
        ":set_default_executor",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl:reference_executor",
        "//tensorflow_federated/python/core/impl/executors:eager_executor",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
    ],
)

py_library(
    name = "get_context_stack",
    srcs = ["get_context_stack.py"],
    srcs_version = "PY3",
    deps = [":context_stack_impl"],
)

py_test(
    name = "get_context_stack_test",
    size = "small",
    srcs = ["get_context_stack_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context_stack_impl",
        ":get_context_stack",
        "//tensorflow_federated/python/core/impl:context_stack_test_utils",
        "//tensorflow_federated/python/core/impl:execution_context",
    ],
)

py_library(
    name = "set_default_context",
    srcs = ["set_default_context.py"],
    srcs_version = "PY3",
    deps = [":context_stack_impl"],
)

py_test(
    name = "set_default_context_test",
    size = "small",
    srcs = ["set_default_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":get_context_stack",
        ":set_default_context",
        "//tensorflow_federated/python/core/impl:context_stack_test_utils",
        "//tensorflow_federated/python/core/impl:execution_context",
    ],
)

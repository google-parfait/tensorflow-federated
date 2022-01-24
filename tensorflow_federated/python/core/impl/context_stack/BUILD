load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":context_stack_packages",
    "//tensorflow_federated/experimental/python/core/impl/jax_context:jax_context_packages",
    "//tensorflow_federated/python/core/impl:impl_visibility",
    "//tensorflow_federated/python/core/impl/computation:computation_packages",
    "//tensorflow_federated/python/core/impl/execution_contexts:execution_contexts_packages",
    "//tensorflow_federated/python/core/impl/executor_stacks:executor_stacks_packages",
    "//tensorflow_federated/python/core/impl/executors:executors_packages",
    "//tensorflow_federated/python/core/impl/federated_context:federated_context_packages",
    "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_context_packages",
    "//tensorflow_federated/python/core/impl/wrappers:wrappers_packages",
    "//tensorflow_federated/python/core/impl/xla_context:xla_context_packages",
])

package_group(
    name = "context_stack_packages",
    packages = ["//tensorflow_federated/python/core/impl/context_stack/..."],
)

licenses(["notice"])

py_library(
    name = "context_stack",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "context_base",
    srcs = ["context_base.py"],
    srcs_version = "PY3",
)

py_library(
    name = "context_stack_base",
    srcs = ["context_stack_base.py"],
    srcs_version = "PY3",
)

py_library(
    name = "context_stack_impl",
    srcs = ["context_stack_impl.py"],
    srcs_version = "PY3",
    deps = [
        ":context_base",
        ":context_stack_base",
        ":runtime_error_context",
        "//tensorflow_federated/python/common_libs:py_typecheck",
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
        ":context_stack_test_utils",
    ],
)

py_library(
    name = "context_stack_test_utils",
    srcs = ["context_stack_test_utils.py"],
    srcs_version = "PY3",
    deps = [":context_base"],
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
    ],
)

py_library(
    name = "runtime_error_context",
    srcs = ["runtime_error_context.py"],
    srcs_version = "PY3",
    deps = [":context_base"],
)

py_library(
    name = "set_default_context",
    srcs = ["set_default_context.py"],
    srcs_version = "PY3",
    deps = [
        ":context_stack_impl",
        ":runtime_error_context",
    ],
)

py_test(
    name = "set_default_context_test",
    size = "small",
    srcs = ["set_default_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":context_stack_impl",
        ":context_stack_test_utils",
        ":set_default_context",
    ],
)

py_library(
    name = "symbol_binding_context",
    srcs = ["symbol_binding_context.py"],
    srcs_version = "PY3",
    deps = [":context_base"],
)

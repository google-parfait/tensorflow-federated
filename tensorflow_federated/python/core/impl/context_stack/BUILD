load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/core/impl:impl_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/core/impl/context_stack/...",

        # Impl Dependencies
        "//tensorflow_federated/python/core/impl/computation/...",
        "//tensorflow_federated/python/core/impl/executors/...",
        "//tensorflow_federated/python/core/impl/federated_context/...",
        "//tensorflow_federated/python/core/impl/tensorflow_context/...",
        "//tensorflow_federated/python/core/impl/wrappers/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "context_stack",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
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
        "@absl_py//absl/testing:absltest",
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
        "@absl_py//absl/testing:absltest",
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
        "@absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "symbol_binding_context",
    srcs = ["symbol_binding_context.py"],
    srcs_version = "PY3",
    deps = [":context_base"],
)

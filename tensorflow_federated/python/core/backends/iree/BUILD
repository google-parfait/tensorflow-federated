load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [":iree_packages"])

package_group(
    name = "iree_packages",
    packages = ["//tensorflow_federated/python/core/backends/iree/..."],
)

licenses(["notice"])

py_library(
    name = "iree",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "backend_info",
    srcs = ["backend_info.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "backend_info_test",
    srcs = ["backend_info_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("absl-py"),
        ":backend_info",
    ],
)

py_library(
    name = "compiler",
    srcs = ["compiler.py"],
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        ":computation_module",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_test(
    name = "compiler_test",
    srcs = ["compiler_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("tf-nightly"),
        ":backend_info",
        ":compiler",
        ":computation_module",
        ":runtime",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "computation_module",
    srcs = ["computation_module.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_test(
    name = "computation_module_test",
    srcs = ["computation_module_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("tf-nightly"),
        ":computation_module",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "executor",
    srcs = ["executor.py"],
    srcs_version = "PY3",
    deps = [
        requirement("numpy"),
        ":backend_info",
        ":compiler",
        ":runtime",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/executors:executor_base",
        "//tensorflow_federated/python/core/impl/executors:executor_utils",
        "//tensorflow_federated/python/core/impl/executors:executor_value_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_test(
    name = "executor_test",
    srcs = ["executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("tf-nightly"),
        ":backend_info",
        ":executor",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/context_stack:set_default_context",
        "//tensorflow_federated/python/core/impl/execution_contexts:sync_execution_context",
        "//tensorflow_federated/python/core/impl/executors:executor_stacks",
    ],
)

py_library(
    name = "runtime",
    srcs = ["runtime.py"],
    srcs_version = "PY3",
    deps = [
        ":backend_info",
        ":computation_module",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_test(
    name = "runtime_test",
    srcs = ["runtime_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("tf-nightly"),
        ":backend_info",
        ":computation_module",
        ":runtime",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

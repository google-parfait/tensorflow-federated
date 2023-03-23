load("//tensorflow_federated/tools:build_defs.bzl", "py_cpu_gpu_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":executor_stacks_packages",
        "//tensorflow_federated/python/core/impl:impl_users",
        "//tensorflow_federated/python/core/impl/execution_contexts:execution_contexts_packages",
    ],
)

package_group(
    name = "executor_stacks_packages",
    packages = ["//tensorflow_federated/python/core/impl/executor_stacks/..."],
)

licenses(["notice"])

py_library(
    name = "executor_stacks",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "cpp_executor_factory",
    srcs = ["cpp_executor_factory.py"],
    srcs_version = "PY3",
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":executor_stack_bindings",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/executors:cpp_to_python_executor",
        "//tensorflow_federated/python/core/impl/executors:executor_base",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
        "//tensorflow_federated/python/core/impl/executors:executors_errors",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "cpp_executor_factory_test",
    srcs = ["cpp_executor_factory_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":cpp_executor_factory",
        "//tensorflow_federated/python/core/impl/executors:executor_base",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "executor_factory",
    srcs = ["executor_factory.py"],
    data = ["//tensorflow_federated/data:worker_binary"],
    tags = ["ignore_for_dep=third_party.py.IPython.get_ipython"],
    deps = [
        ":python_executor_stacks",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
        "//tensorflow_federated/python/core/impl/executors:remote_executor",
        "//tensorflow_federated/python/core/impl/executors:remote_executor_grpc_stub",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "executor_stack_bindings",
    srcs = ["executor_stack_bindings.py"],
    srcs_version = "PY3",
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        "//tensorflow_federated/python/core/impl/executors:data_conversions",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "executor_stack_bindings_test",
    srcs = ["executor_stack_bindings_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":executor_stack_bindings",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_library(
    name = "python_executor_stacks",
    srcs = ["python_executor_stacks.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:deprecation",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/compiler:local_computation_factory_base",
        "//tensorflow_federated/python/core/impl/compiler:tensorflow_computation_factory",
        "//tensorflow_federated/python/core/impl/executors:eager_tf_executor",
        "//tensorflow_federated/python/core/impl/executors:executor_base",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
        "//tensorflow_federated/python/core/impl/executors:executors_errors",
        "//tensorflow_federated/python/core/impl/executors:federated_composing_strategy",
        "//tensorflow_federated/python/core/impl/executors:federated_resolving_strategy",
        "//tensorflow_federated/python/core/impl/executors:federating_executor",
        "//tensorflow_federated/python/core/impl/executors:reference_resolving_executor",
        "//tensorflow_federated/python/core/impl/executors:remote_executor",
        "//tensorflow_federated/python/core/impl/executors:remote_executor_grpc_stub",
        "//tensorflow_federated/python/core/impl/executors:remote_executor_stub",
        "//tensorflow_federated/python/core/impl/executors:sequence_executor",
        "//tensorflow_federated/python/core/impl/executors:thread_delegating_executor",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_cpu_gpu_test(
    name = "python_executor_stacks_test",
    size = "small",
    srcs = ["python_executor_stacks_test.py"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    deps = [
        ":python_executor_stacks",
        "//tensorflow_federated/python/core/impl/executors:eager_tf_executor",
        "//tensorflow_federated/python/core/impl/executors:executor_base",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
        "//tensorflow_federated/python/core/impl/executors:executor_test_utils",
        "//tensorflow_federated/python/core/impl/executors:federated_composing_strategy",
        "//tensorflow_federated/python/core/impl/executors:federating_executor",
        "//tensorflow_federated/python/core/impl/executors:remote_executor",
        "//tensorflow_federated/python/core/impl/executors:remote_executor_grpc_stub",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/tensorflow_libs:tensorflow_test_utils",
    ],
)

py_test(
    name = "python_executor_stacks_multi_gpu_test",
    srcs = ["python_executor_stacks_multi_gpu_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":python_executor_stacks",
        "//tensorflow_federated/python/core/impl/executors:executor_base",
        "//tensorflow_federated/python/core/impl/executors:executor_test_utils",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/tensorflow_libs:tensorflow_test_utils",
    ],
)

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
    visibility = ["//tools/python_package:python_package_tool"],
)

py_library(
    name = "cpp_executor_factory",
    srcs = ["cpp_executor_factory.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":executor_stack_bindings",
        "@federated_language//federated_language",
        "@federated_language_executor//federated_language_executor",
    ],
)

py_test(
    name = "cpp_executor_factory_test",
    srcs = ["cpp_executor_factory_test.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":cpp_executor_factory",
        "@federated_language//federated_language",
        "@federated_language_executor//federated_language_executor",
        "@federated_language_executor//federated_language_executor:executor_test_utils_bindings",  # build_cleaner: keep
    ],
)

py_library(
    name = "executor_factory",
    srcs = ["executor_factory.py"],
    data = ["//tensorflow_federated/data:worker_binary"],
    tags = ["ignore_for_dep=third_party.py.IPython.get_ipython"],
    deps = [
        ":python_executor_stacks",
        "//tensorflow_federated/python/core/impl/executors:remote_executor",
        "//tensorflow_federated/python/core/impl/executors:remote_executor_grpc_stub",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "executor_stack_bindings",
    srcs = ["executor_stack_bindings.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        "//tensorflow_federated/cc/core/impl/executor_stacks:executor_stack_bindings",
        "@federated_language//federated_language",
        "@federated_language_executor//federated_language_executor",
    ],
)

py_test(
    name = "executor_stack_bindings_test",
    srcs = ["executor_stack_bindings_test.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":executor_stack_bindings",
        "@federated_language//federated_language",
        "@federated_language_executor//federated_language_executor",
    ],
)

py_library(
    name = "python_executor_stacks",
    srcs = ["python_executor_stacks.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "python_executor_stacks_test",
    size = "small",
    srcs = ["python_executor_stacks_test.py"],
    shard_count = 5,
    deps = [
        ":python_executor_stacks",
        "@federated_language//federated_language",
    ],
)

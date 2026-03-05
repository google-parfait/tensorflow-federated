load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":test_packages",
        "//tensorflow_federated/python/core:core_users",

        # TODO: b/233801300 - Remove dependency from `mapreduce` to `test`.
        "//tensorflow_federated/python/core/backends/mapreduce:mapreduce_packages",
    ],
)

package_group(
    name = "test_packages",
    packages = ["//tensorflow_federated/python/core/backends/test/..."],
)

licenses(["notice"])

py_library(
    name = "test",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/core/backends:__pkg__"],
    deps = [
        ":compiler",
        ":execution_contexts",
    ],
)

py_library(
    name = "compiler",
    srcs = ["compiler.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/backends/mapreduce:intrinsics",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_building_block_factory",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_computation_factory",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_tree_transformations",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "compiler_test",
    srcs = ["compiler_test.py"],
    deps = [
        ":compiler",
        "//tensorflow_federated/python/core/backends/mapreduce:intrinsics",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_tree_transformations",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "cpp_execution_contexts",
    srcs = ["cpp_execution_contexts.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":compiler",
        "//tensorflow_federated/python/core/backends/native:compiler",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_executor_bindings",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/executor_stacks:cpp_executor_factory",
        "//tensorflow_federated/python/core/impl/executors:executor_bindings",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "cpp_execution_contexts_test",
    srcs = ["cpp_execution_contexts_test.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":cpp_execution_contexts",
        ":execution_contexts",
        "//tensorflow_federated/python/core/backends/mapreduce:intrinsics",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "execution_contexts",
    srcs = ["execution_contexts.py"],
    deps = [
        ":compiler",
        "//tensorflow_federated/python/core/backends/native:compiler",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/executor_stacks:executor_factory",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "execution_contexts_test",
    srcs = ["execution_contexts_test.py"],
    deps = [
        ":execution_contexts",
        "//tensorflow_federated/python/core/backends/mapreduce:intrinsics",
        "@federated_language//federated_language",
    ],
)

load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [":framework_packages"],
)

package_group(
    name = "framework_packages",
    packages = ["//tensorflow_federated/python/core/framework/..."],
)

licenses(["notice"])

py_library(
    name = "framework",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/computation:computation_serialization",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/context_stack:get_context_stack",
        "//tensorflow_federated/python/core/impl/context_stack:set_default_context",
        "//tensorflow_federated/python/core/impl/execution_contexts:async_execution_context",
        "//tensorflow_federated/python/core/impl/execution_contexts:mergeable_comp_execution_context",
        "//tensorflow_federated/python/core/impl/execution_contexts:sync_execution_context",
        "//tensorflow_federated/python/core/impl/executor_stacks:executor_factory",
        "//tensorflow_federated/python/core/impl/executors:cardinalities_utils",
        "//tensorflow_federated/python/core/impl/executors:cardinality_carrying_base",
        "//tensorflow_federated/python/core/impl/executors:data_descriptor",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
        "//tensorflow_federated/python/core/impl/executors:executors_errors",
        "//tensorflow_federated/python/core/impl/executors:ingestable_base",
        "//tensorflow_federated/python/core/impl/executors:remote_executor_stub",
        "//tensorflow_federated/python/core/impl/executors:value_serialization",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

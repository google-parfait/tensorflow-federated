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
        "//tensorflow_federated/python/core/impl/compiler:transformations",
        "//tensorflow_federated/python/core/impl/execution_contexts:mergeable_comp_execution_context",
        "//tensorflow_federated/python/core/impl/executor_stacks:executor_factory",
        "//tensorflow_federated/python/core/impl/executor_stacks:python_executor_stacks",
        "//tensorflow_federated/python/core/impl/executors:remote_executor",
        "//tensorflow_federated/python/core/impl/executors:remote_executor_grpc_stub",
        "//tensorflow_federated/python/core/impl/executors:remote_executor_stub",
    ],
)

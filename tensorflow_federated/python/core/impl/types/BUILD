load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":types_packages",
        "//tensorflow_federated/python/core/impl:impl_users",
        "//tensorflow_federated/python/core/impl/compiler:compiler_packages",
        "//tensorflow_federated/python/core/impl/execution_contexts:execution_contexts_packages",
        "//tensorflow_federated/python/core/impl/executor_stacks:executor_stacks_packages",
        "//tensorflow_federated/python/core/impl/executors:executors_packages",
    ],
)

package_group(
    name = "types_packages",
    packages = ["//tensorflow_federated/python/core/impl/types/..."],
)

licenses(["notice"])

py_library(
    name = "types",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = ["@federated_language//federated_language"],
)

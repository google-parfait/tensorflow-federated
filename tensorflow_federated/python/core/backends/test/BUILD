load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":test_packages",
    "//tensorflow_federated/python/core:core_visibility",

    # TODO(b/233801300): Remove dependency from `mapreduce` to `native`.
    "//tensorflow_federated/python/core/backends/mapreduce:mapreduce_packages",
])

package_group(
    name = "test_packages",
    packages = ["//tensorflow_federated/python/core/backends/test/..."],
)

licenses(["notice"])

py_library(
    name = "test",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/core/backends:__pkg__"],
    deps = [":execution_contexts"],
)

py_library(
    name = "execution_contexts",
    srcs = ["execution_contexts.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_reductions",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/execution_contexts:sync_execution_context",
        "//tensorflow_federated/python/core/impl/executors:executor_stacks",
    ],
)

py_test(
    name = "execution_contexts_test",
    srcs = ["execution_contexts_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

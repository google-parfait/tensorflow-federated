load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":executor_stacks_packages",
    "//tensorflow_federated/python/core/impl:impl_visibility",
])

package_group(
    name = "executor_stacks_packages",
    packages = ["//tensorflow_federated/python/core/impl/executor_stacks/..."],
)

licenses(["notice"])

py_library(
    name = "executor_stacks",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:private"],
)

py_library(
    name = "cpp_executor_factory",
    srcs = ["cpp_executor_factory.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_stack_bindings",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/executors:executor_bindings",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@absl_py//absl/logging",
    ],
)

py_test(
    name = "cpp_executor_factory_test",
    srcs = ["cpp_executor_factory_test.py"],
    data = ["@pybind11_abseil//pybind11_abseil:status.so"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cpp_executor_factory",
        "//tensorflow_federated/python/core/impl/executors:executor_bindings",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",  # build_cleaner: keep
    ],
)

py_library(
    name = "executor_stack_bindings",
    srcs = ["executor_stack_bindings.py"],
    data = ["//tensorflow_federated/cc/core/impl/executor_stacks:executor_stack_bindings.so"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/impl/executors:data_conversions",
        "//tensorflow_federated/python/core/impl/executors:executor_bindings",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@com_google_protobuf//:protobuf_python",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "executor_stack_bindings_test",
    srcs = ["executor_stack_bindings_test.py"],
    data = ["@pybind11_abseil//pybind11_abseil:status.so"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":executor_stack_bindings",
        "//tensorflow_federated/python/core/impl/executors:executor_bindings",
        "//tensorflow_federated/python/core/impl/types:placements",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

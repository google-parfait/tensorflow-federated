load("@org_tensorflow//third_party/xla/third_party/py/rules_pywrap:pywrap.impl.bzl", "pywrap_library")
load("@rules_python//python:defs.bzl", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

package_group(
    name = "impl_packages",
    packages = [
        "//tensorflow_federated/cc/core/impl/...",
        "//tensorflow_federated/python/core/impl/...",
    ],
)

package_group(
    name = "impl_users",
    includes = [
        "//tensorflow_federated/cc/simulation:simulation_packages",
        "//tensorflow_federated/python/core/impl:impl_users",
    ],
)

licenses(["notice"])

pywrap_library(
    name = "tff_bindings",
    common_lib_version_scripts = {
        "tensorflow_federated/cc/core/impl/tff_bindings_common": ":tff_bindings.lds",
    },
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_federated/cc/core/impl/executor_stacks:executor_stack_bindings",
        "//tensorflow_federated/cc/core/impl/executors:executor_bindings",
        "//tensorflow_federated/cc/core/impl/executors:tensorflow_bindings",
    ],
)

pywrap_library(
    name = "tff_test_bindings",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_federated/cc/core/impl/executors:executor_test_utils_bindings",
    ],
)

py_library(
    name = "cc_impl_init",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
)

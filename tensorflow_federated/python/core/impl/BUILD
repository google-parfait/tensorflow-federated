load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:private"])

package_group(
    name = "impl_packages",
    packages = [
        "//tensorflow_federated/experimental/python/core/impl/...",
        "//tensorflow_federated/python/core/impl/...",
    ],
)

package_group(
    name = "impl_visibility",
    includes = [
        "//tensorflow_federated/python/aggregators:aggregators_packages",
        "//tensorflow_federated/python/analytics:analytics_packages",
        "//tensorflow_federated/python/core/api:api_packages",
        "//tensorflow_federated/python/core/backends:backends_packages",
        "//tensorflow_federated/python/core/framework:framework_packages",
        "//tensorflow_federated/python/core/templates:templates_packages",
        "//tensorflow_federated/python/core/test:test_packages",
        "//tensorflow_federated/python/core/types:types_packages",
        "//tensorflow_federated/python/learning:learning_packages",
        "//tensorflow_federated/python/simulation:simulation_packages",
    ],
    packages = [
        # Namespace visibility
        "//tensorflow_federated",
    ],
)

licenses(["notice"])

py_library(
    name = "impl",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

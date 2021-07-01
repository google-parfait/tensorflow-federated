load("@rules_python//python:defs.bzl", "py_library")

# Packages that have visibility into the `core` package.
package_group(
    name = "core_visibility",
    includes = [
        "//tensorflow_federated/python/learning:learning_visibility",
        "//tensorflow_federated/python/simulation:simulation_visibility",
    ],
    packages = [
        # TFF dependencies
        "//tensorflow_federated/python/aggregators/...",
        "//tensorflow_federated/python/learning/...",
        "//tensorflow_federated/python/simulation/...",
        "//tensorflow_federated/python/analytics/...",

        # Other non-learning federated algorithm dependencies
        "//tensorflow_federated/experimental/...",
    ],
)

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "core",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

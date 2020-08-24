# Direct and indirect packages that have visibility into the `core` package.
package_group(
    name = "core_visibility",
    includes = [
        "//tensorflow_federated/python/simulation:simulation_visibility",
    ],
    packages = [
        # TFF Dependencies
        "//tensorflow_federated/python/aggregators/...",
        "//tensorflow_federated/python/learning/...",
        "//tensorflow_federated/python/simulation/...",

        # Other non-learning federated algorithm dependencies
        "//tensorflow_federated/python/core/utils/...",
    ],
)

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "core",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow_federated/tools:__subpackages__",
    ],
)

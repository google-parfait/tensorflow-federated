load("@rules_python//python:defs.bzl", "py_library", "py_test")

# Packages that have visibility into the `simulation` package.
package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/simulation:simulation_visibility"],
    packages = [
        "//tensorflow_federated/python/simulation/train/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "train",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/simulation:__pkg__"],
    deps = [
        ":checkpoint_manager",
        ":csv_manager",
        ":metrics_manager",
    ],
)

py_library(
    name = "checkpoint_manager",
    srcs = ["checkpoint_manager.py"],
    srcs_version = "PY3",
)

py_test(
    name = "checkpoint_manager_test",
    srcs = ["checkpoint_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":checkpoint_manager"],
)

py_library(
    name = "csv_manager",
    srcs = ["csv_manager.py"],
    srcs_version = "PY3",
    deps = [":metrics_manager"],
)

py_test(
    name = "csv_manager_test",
    srcs = ["csv_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":csv_manager"],
)

py_library(
    name = "metrics_manager",
    srcs = ["metrics_manager.py"],
    srcs_version = "PY3",
)

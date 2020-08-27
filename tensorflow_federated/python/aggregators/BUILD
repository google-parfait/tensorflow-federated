package_group(
    name = "default_visibility",
    packages = ["//tensorflow_federated/python/aggregators/..."],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "aggregators",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated:__pkg__"],
)

py_library(
    name = "factory",
    srcs = ["factory.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/templates:aggregation_process",
    ],
)

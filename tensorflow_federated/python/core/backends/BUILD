package(default_visibility = [
    # Subpackages
    "//tensorflow_federated/python/core/backends:__subpackages__",
])

licenses(["notice"])

py_library(
    name = "backends",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow_federated:__pkg__",
        "//tensorflow_federated/python/core:__pkg__",
    ],
    deps = [
        "//tensorflow_federated/python/core/backends/mapreduce",
        "//tensorflow_federated/python/core/backends/native",
    ],
)

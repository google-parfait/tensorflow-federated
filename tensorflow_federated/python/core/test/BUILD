package(default_visibility = [
    # Subpackages
    "//tensorflow_federated/python/core/testing:__subpackages__",
])

licenses(["notice"])

py_library(
    name = "test",
    srcs = ["__init__.py"],
    visibility = [
        "//tensorflow_federated:__pkg__",
        "//tensorflow_federated/python/core:__pkg__",
    ],
    deps = [
        "//tensorflow_federated/python/core/impl:reference_executor",
    ],
)

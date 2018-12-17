package(default_visibility = ["//tensorflow_federated/python/learning:__subpackages__"])

licenses(["notice"])  # Apache 2.0

# This is the import for clients that contribute components to federated learning (see `../README.md`).
py_library(
    name = "framework",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_federated/python/learning:federated_averaging",
    ],
)

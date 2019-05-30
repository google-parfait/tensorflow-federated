package(default_visibility = ["//tensorflow_federated:__subpackages__"])

licenses(["notice"])  # Apache 2.0

# Modules outside the core/ directory should import this.
py_library(
    name = "core",
    srcs = ["__init__.py"],
    deps = [
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:typed_object",
        "//tensorflow_federated/python/core/api:value_base",
        "//tensorflow_federated/python/core/api:values",
        "//tensorflow_federated/python/core/framework",
        "//tensorflow_federated/python/core/utils",
    ],
)

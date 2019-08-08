package(default_visibility = ["//tensorflow_federated/python/core:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "framework",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl:computation_building_block_utils",
        "//tensorflow_federated/python/core/impl:computation_constructing_utils",
        "//tensorflow_federated/python/core/impl:computation_wrapper_instances",
        "//tensorflow_federated/python/core/impl:intrinsic_defs",
        "//tensorflow_federated/python/core/impl:intrinsic_reductions",
        "//tensorflow_federated/python/core/impl:transformation_utils",
        "//tensorflow_federated/python/core/impl:transformations",
        "//tensorflow_federated/python/core/impl:tree_analysis",
        "//tensorflow_federated/python/core/impl:type_utils",
    ] + select({
        "//tensorflow_federated:py2_mode": [],
        "//tensorflow_federated:py3_mode": [
            "//tensorflow_federated/python/core/impl:caching_executor",
            "//tensorflow_federated/python/core/impl:concurrent_executor",
            "//tensorflow_federated/python/core/impl:eager_executor",
            "//tensorflow_federated/python/core/impl:executor_base",
            "//tensorflow_federated/python/core/impl:executor_service",
            "//tensorflow_federated/python/core/impl:executor_stacks",
            "//tensorflow_federated/python/core/impl:executor_value_base",
            "//tensorflow_federated/python/core/impl:federated_executor",
            "//tensorflow_federated/python/core/impl:lambda_executor",
            "//tensorflow_federated/python/core/impl:remote_executor",
            "//tensorflow_federated/python/core/impl:set_default_executor",
            "//tensorflow_federated/python/core/impl:transforming_executor",
        ],
    }),
)

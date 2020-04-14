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
        ":static_assert",
        "//tensorflow_federated/python/core/impl:reference_executor",
    ],
)

py_library(
    name = "static_assert",
    srcs = ["static_assert.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
    ],
)

py_test(
    name = "static_assert_test",
    size = "small",
    srcs = ["static_assert_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":static_assert",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
    ],
)

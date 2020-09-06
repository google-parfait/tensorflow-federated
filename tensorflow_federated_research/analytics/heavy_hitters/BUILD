package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "heavy_hitters_utils",
    srcs = ["heavy_hitters_utils.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated_research"],
    deps = ["//tensorflow_federated"],
)

py_library(
    name = "heavy_hitters_testcase",
    testonly = True,
    srcs = ["heavy_hitters_testcase.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated_research"],
)

py_test(
    name = "heavy_hitters_utils_test",
    srcs = ["heavy_hitters_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["manual"],
    deps = [":heavy_hitters_utils"],
)

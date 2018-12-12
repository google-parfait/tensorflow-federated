# Placeholder for internal Python version compatibility macro.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "encoding_stage",
    srcs = ["encoding_stage.py"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "encoding_stage_test",
    srcs = ["encoding_stage_test.py"],
    deps = [
        ":encoding_stage",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
